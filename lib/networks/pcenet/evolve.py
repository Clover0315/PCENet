import torch.nn as nn
from .snake import Snake
from lib.utils.snake import snake_gcn_utils, snake_config, snake_decode, active_spline, snake_gcn_whu_utils
import torch
from lib.utils import data_utils


class Evolution(nn.Module):
    def __init__(self):
        super(Evolution, self).__init__()

        self.fuse = nn.Conv1d(128, 64, 1)
        # # 获取外接多边形
        # self.rect_iter = 2
        # for i in range(self.rect_iter):
        #     rect_gcn = Snake(state_dim=128, feature_dim=64 + 2, conv_type='dgrid', dilation=[1, 1, 1, 2, 2, 4, 4])
        #     self.__setattr__('rect_gcn' + str(i), rect_gcn)

        # 初始化
        # self.init_gcn = Snake(state_dim=128, feature_dim=64 + 2, conv_type='dgrid')
        self.init_iter = 3
        for i in range(self.init_iter):
            init_gcn = Snake(feature_dim=64 + 2, conv_type='dgrid')
            self.__setattr__('init_gcn' + str(i), init_gcn)

        # 演化
        self.evolve_gcn = Snake(feature_dim=64 + 2, conv_type='dgrid')
        self.iter = 1
        for i in range(self.iter):
            evolve_gcn = Snake(feature_dim=64 + 2, conv_type='dgrid')
            self.__setattr__('evolve_gcn' + str(i), evolve_gcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prepare_training(self, output, batch):
        init = snake_gcn_utils.prepare_training(output, batch)
        output.update({
            'i_it_4py': init['i_it_4py'],
        })
        output.update({
            'i_gt_4py': init['i_gt_4py'],
            'i_gt_py': init['i_gt_py']
        })
        if 'gt_pys' in init:
            output.update({
                'gt_pys': init['gt_pys'],
                'gt_pys_num': init['gt_pys_num']
            })
        return init

    def prepare_testing_init(self, output):
        init = snake_gcn_whu_utils.prepare_testing_init(output['detection'][..., :4], output['detection'][..., 4])
        output['detection'] = output['detection'][output['detection'][..., 4] > snake_config.ct_score]
        output.update({'it_ex': init['i_it_4py']})
        return init

    def prepare_testing_evolve(self, output, h, w):
        ex = output['ex']  # extreme point 极值点
        # 端点限制在图片边界内
        ex[..., 0] = torch.clamp(ex[..., 0], min=0, max=w - 1)
        ex[..., 1] = torch.clamp(ex[..., 1], min=0, max=h - 1)
        # 采样为128个点
        evolve = snake_gcn_whu_utils.prepare_testing_evolve(ex)
        output.update({'it_py': evolve['i_it_py']})
        return evolve

    def init_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind,
                  skip_num=0):
        if len(i_it_poly) == 0:
            return torch.zeros([0, 4, 2]).to(i_it_poly)

        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)  # Tensor（N,64,40）

        # 融合中心点热力图
        center = (torch.min(i_it_poly, dim=1)[0] + torch.max(i_it_poly, dim=1)[0]) * 0.5
        ct_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, center[:, None], ind, h, w)
        init_feature = torch.cat([init_feature, ct_feature.expand_as(init_feature)], dim=1)
        init_feature = self.fuse(init_feature)

        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        # adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device)
        # i_poly = i_it_poly + snake(init_input).permute(0, 2, 1)

        # x,y拆开
        res = snake(init_input)
        diff_x, diff_y = res[0], res[1]
        i_poly_x = i_it_poly[..., 0] + torch.squeeze(diff_x)
        i_poly_y = i_it_poly[..., 1] + torch.squeeze(diff_y)
        i_poly = torch.stack([i_poly_x, i_poly_y], dim=-1)

        if skip_num:
            i_poly = i_poly[:, ::skip_num]  # 每skip_num个取一个，只计算这几个点的损失

        return i_poly

    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind, return_state=False):
        if len(i_it_poly) == 0:
            if not return_state:
                return torch.zeros_like(i_it_poly)
            else:
                return torch.zeros_like(i_it_poly), None
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)

        c_it_poly = c_it_poly * snake_config.ro  # 扩大了四倍
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        # adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device)
        # i_poly = i_it_poly * snake_config.ro + snake(init_input).permute(0, 2, 1)

        # x,y拆开
        res = snake(init_input, return_state)
        diff_x, diff_y = res[0], res[1]
        i_poly_x = i_it_poly[..., 0] * snake_config.ro + torch.squeeze(diff_x)
        i_poly_y = i_it_poly[..., 1] * snake_config.ro + torch.squeeze(diff_y)
        i_poly = torch.stack([i_poly_x, i_poly_y], dim=-1)

        if not return_state:
            return i_poly
        else:
            return i_poly, res[2]

    def can_poly(self, poly):
        if not snake_config.pan_center:
            return snake_gcn_utils.img_poly_to_can_poly(poly)
        else:
            return snake_gcn_whu_utils.img_poly_to_center_poly(poly)

    def forward(self, output, cnn_feature, batch=None):
        ret = output

        if batch is not None and 'test' not in batch['meta']:
            with torch.no_grad():
                init = self.prepare_training(output, batch)

            # snake推理init
            # ex_pred = self.init_poly(self.init_gcn, cnn_feature, init['i_it_min_4py'],
            #                          init['c_it_min_4py'], init['4py_ind'])
            # 迭代 init
            ex_pred = None
            init_poly = init['i_it_4py']
            can_init = init['c_it_4py']
            for i in range(self.init_iter):
                init_gcn = self.__getattr__('init_gcn' + str(i))
                ex_pred = self.init_poly(init_gcn, cnn_feature, init_poly, can_init, init['4py_ind'])
                init_poly = ex_pred
                can_init = self.can_poly(init_poly)

            ret.update({'ex_pred': ex_pred, 'i_gt_4py': output['i_gt_4py']})
            # 外接矩形演化为最小外接矩形
            # min_area_pred = None
            # init_rect = init['i_it_min_4py']
            # can_rect = init['c_it_min_4py']
            # for i in range(self.rect_iter):
            #     rect_gcn = self.__getattr__('rect_gcn' + str(i))
            #     min_area_pred = self.init_poly(rect_gcn, cnn_feature, init_rect,
            #                                    can_rect, init['4py_ind'], snake_config.rect_poly_num // 4)
            #     init_rect = snake_gcn_utils.uniform_upsample(min_area_pred[None], snake_config.rect_poly_num)[0]
            #     # init_rect = min_area_pred
            #     can_rect = self.can_poly(init_rect)
            #
            # # min_area_pred = min_area_pred[:, ::snake_config.rect_poly_num // 4]
            # ret.update({'min_area_pred': min_area_pred, 'i_gt_min_4py': output['i_gt_min_4py']})

            # 最小外接矩形演化至外接多边形
            # ex_pred = self.evolve_poly(self.init_gcn, cnn_feature, init['i_it_4py'],
            #                          init['c_it_4py'], init['4py_ind'])
            # ex_pred = ex_pred / snake_config.ro
            # ret.update({'ex_pred': ex_pred, 'i_gt_4py': output['i_gt_4py']})

            # snake推理py
            i_it_pys = snake_gcn_utils.uniform_upsample(ex_pred[None], snake_config.poly_num)[0]
            c_it_pys = snake_gcn_whu_utils.can_poly(i_it_pys)

            py_pred = self.evolve_poly(self.evolve_gcn, cnn_feature, i_it_pys, c_it_pys, init['py_ind'])
            py_preds = [py_pred]
            # 迭代演化
            state = None
            for i in range(self.iter):
                py_pred = py_pred / snake_config.ro
                c_py_pred = self.can_poly(py_pred)
                evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
                return_state = True if i == self.iter - 1 else False
                py_pred, state = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred, init['py_ind'], return_state)
                py_preds.append(py_pred)
            ret.update({
                'py_pred': py_preds, 'i_gt_py': output['i_gt_py'] * snake_config.ro,
                'state': state
            })

        if not self.training:
            with torch.no_grad():
                init = self.prepare_testing_init(output)
                # 通过snake获得ex
                # ex = self.init_poly(self.init_gcn, cnn_feature, init['i_it_4py'], init['c_it_4py'],
                #                     init['ind'])

                # rect = snake_gcn_utils.uniform_upsample(min_area_pred[None], snake_config.init_poly_num)[0]
                # ex = self.evolve_poly(self.init_gcn, cnn_feature, rect,
                #                       self.can_poly(rect), init['ind'])
                # ex = ex / snake_config.ro
                # ret.update({'ex': ex})
                # evolve = self.prepare_testing_evolve(output, cnn_feature.size(2), cnn_feature.size(3))
                # py = self.evolve_poly(self.evolve_gcn, cnn_feature, evolve['i_it_py'], evolve['c_it_py'], init['ind'],)
                # pys = [py / snake_config.ro]
                # for i in range(self.iter):
                #     py = py / snake_config.ro
                #     c_py = self.can_poly(py)  # 多边形平移到边缘
                #     evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
                #     py = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['ind'])
                #     pys.append(py / snake_config.ro)
                # ret.update({'py': pys})

                ret.update({'ex': ex_pred})
                ret.update({'py': [py / snake_config.ro for py in py_preds]})
        return output

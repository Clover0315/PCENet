import torch
import torch.nn as nn
from ..snake3.snake import Prediction

from lib.utils.snake import snake_gcn_utils, snake_config, snake_decode, active_spline, snake_gcn_whu_utils


class EndpointClass(nn.Module):
    def __init__(self, state_dim=snake_config.state_dim, res_layer_num=7, fusion_state_dim=256):
        super(EndpointClass, self).__init__()
        self.ep_cls = Prediction()
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def ep_classification(self, cnn_feature, i_it_poly):
        if len(i_it_poly) == 0:
            # return torch.zeros([0, snake_config.poly_num, 2]).to(cnn_feature)
            return torch.zeros([0, snake_config.poly_num]).to(cnn_feature)

        # snake获得分类
        endpoint_type = self.ep_cls(cnn_feature).permute(0, 2, 1)
        endpoint_type = torch.squeeze(endpoint_type, dim=2)

        return endpoint_type

    def prepare_training(self, output, batch):
        init = snake_gcn_whu_utils.prepare_training_ep_tp(output, batch)
        output.update({
            'gt_pys': init['gt_pys'],
            'gt_pys_num': init['gt_pys_num']
        })
        return init

    def forward(self, output, batch=None):
        ret = output
        if batch is not None and 'test' not in batch['meta']:
            with torch.no_grad():
                init = self.prepare_training(output, batch)

            py_pred = output['py_pred'][-1] / snake_config.ro  # feature map缩小了四倍
            cnn_feature = output['state']
            if type(cnn_feature) == torch.Tensor:
                endpoint_type = self.ep_classification(cnn_feature, py_pred)
            else:
                endpoint_type = torch.zeros((py_pred.size(0), py_pred.size(1))).to(py_pred)
            ret.update({
                'init_ep_tp': py_pred,
                'pred_ep_tp': endpoint_type
            })
        return output


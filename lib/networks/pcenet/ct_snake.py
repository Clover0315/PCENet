import numpy as np
import torch.nn as nn
from .dla import DLASeg
from .evolve import Evolution
from lib.utils import net_utils, data_utils
from lib.utils.snake import snake_decode, snake_config, snake_gcn_utils
import torch
from lib.config import cfg


class Network(nn.Module):
    def __init__(self, num_layers, heads, head_conv=256, down_ratio=4):
        super(Network, self).__init__()

        '''
        num_layers dla的层数，这是设置为34，则base网络使用dla34
        heads = {ct_hm, wh}
        '''
        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=False,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv)

        self.gcn = Evolution()

    def decode_detection(self, output, h, w):
        '''
        从ct_hm wh解码获得检测框
        :param output:
        :param h:
        :param w:
        :return:
        '''
        ct_hm = output['ct_hm']
        wh = output['wh']
        ct, detection = snake_decode.decode_ct_hm(torch.sigmoid(ct_hm), wh)  # 生成热力图
        detection[..., :4] = data_utils.clip_to_image(detection[..., :4], h, w)
        output.update({'ct': ct, 'detection': detection})
        return ct, detection

    def use_gt_detection(self, output, batch):
        _, _, height, width = output['ct_hm'].size()
        ct_01 = batch['ct_01'].byte()

        ct_ind = batch['ct_ind'][ct_01]
        xs, ys = ct_ind % width, ct_ind // width
        xs, ys = xs[:, None].float(), ys[:, None].float()
        ct = torch.cat([xs, ys], dim=1)

        wh = batch['wh'][ct_01]
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=1)
        score = torch.ones([len(bboxes)]).to(bboxes)[:, None]
        ct_cls = batch['ct_cls'][ct_01].float()[:, None]
        detection = torch.cat([bboxes, score, ct_cls], dim=1)

        output['ct'] = ct[None]
        output['detection'] = detection[None]

        return output

    def forward(self, x, batch=None, return_cnn=False):
        # cnn_feature Tensor(1, 64, 80, 80)
        # output:{ct_hm Tensor(1,1,80,80) [中心点热力图], wh Tensor(1,2,80,80)}
        output, cnn_feature = self.dla(x)
        with torch.no_grad():
            # detection Tensor (1, 100, 6)[0~4是bbox坐标，第5个是bbox的分数, 第6个是分类]}
            ct, detection = self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))
        if cfg.use_gt_det:
            self.use_gt_detection(output, batch)
        output = self.gcn(output, cnn_feature, batch)

        if return_cnn:
            return output, cnn_feature
        return output

    def freeze_backbone_dla(self):
        for param in self.dla.parameters():
            param.requires_grad = False

    def unfreeze_backbone_dla(self):
        for param in self.dla.parameters():
            param.requires_grad = True


def get_network(num_layers, heads, head_conv=256, down_ratio=4):
    network = Network(num_layers, heads, head_conv, down_ratio)
    return network

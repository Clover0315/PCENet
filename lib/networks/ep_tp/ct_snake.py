import torch.nn as nn
from .endpoint_class import EndpointClass
from lib.utils.snake import snake_gcn_utils, snake_config
import torch
from lib.config import cfg
if cfg.backbone == 'snake':
    from ..snake import get_network as backbone_net
else:
    from ..pcenet import get_network as backbone_net
from ...utils import net_utils


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.backbone = backbone_net(cfg)
        #
        self.ep_tp = EndpointClass()

    def forward(self, x, batch=None):
        output = self.backbone(x, batch)
        endpoint_type = self.ep_tp(output, batch)

        if not self.training:
            # 根据端点类别处理pred
            ep_tps = net_utils.sigmoid(endpoint_type['pred_ep_tp'])
            pys_list = output['py']
            pys = pys_list[-1]
            _py_arr = []
            valid_arr = []
            p_num = pys.shape[0]  # 多边形的数目
            if p_num:
                for i in range(p_num):
                    py = pys[i][ep_tps[i] > snake_config.ep_score, :]
                    if py.shape[0] >= 3:
                        valid_arr.append(py)
                        py = snake_gcn_utils.uniform_upsample(py[None][None], snake_config.poly_num)[0]
                    else:
                        py = torch.zeros_like(pys[i][None])
                    _py_arr.append(py)
                _pys = torch.cat(_py_arr)
                # TODO  切换设置是否evaluate轮廓简化结果
                # pys_list.append(_pys)
                # output.update({'py': pys_list})
                output.update({'valid_py': valid_arr})
        return output

    def freeze_dla(self):
        for param in self.backbone.dla.parameters():
            param.requires_grad = False

    def unfreeze_dla(self):
        for param in self.backbone.dla.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


def get_network():
    network = Network()
    return network


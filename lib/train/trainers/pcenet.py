import torch.nn as nn
from lib.utils import net_utils
from lib.utils.snake import snake_config
import torch


def format_loss(loss_func, pred, gt, **kys):
    '''解决多边形数量为0的情况'''
    if gt.size(0) == 0:
        if pred.size(0) == 0:
            return torch.tensor(0.).to(pred)
        else:
            gt = torch.zeros_like(pred).to(pred)
    else:
        if pred.size(0) == 0:
            pred = torch.zeros_like(gt).to(pred)
    return loss_func(pred, gt, **kys)


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.ct_crit = net_utils.FocalLoss()
        self.wh_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.reg_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.init_crit = torch.nn.functional.smooth_l1_loss
        # self.ex_crit = torch.nn.functional.smooth_l1_loss
        self.ex_crit = net_utils.PolyMinMatchingLoss(snake_config.init_poly_num, loss_type=1)
        # self.py_crit = torch.nn.functional.smooth_l1_loss
        self.py_crit = net_utils.PolyMinMatchingLoss(snake_config.poly_num, loss_type=1)

    def forward(self, batch):
        output = self.net(batch['inp'], batch)

        scalar_stats = {}
        loss = 0

        ct_loss = self.ct_crit(net_utils.sigmoid(output['ct_hm']), batch['ct_hm'])
        scalar_stats.update({'ct_loss': ct_loss})
        loss += ct_loss

        wh_loss = self.wh_crit(output['wh'], batch['wh'], batch['ct_ind'], batch['ct_01'])
        scalar_stats.update({'wh_loss': wh_loss})
        loss += 0.1 * wh_loss

        ex_loss = format_loss(self.ex_crit, output['ex_pred'], output['i_gt_4py'])
        scalar_stats.update({'ex_loss': ex_loss})
        loss += ex_loss

        py_loss = 0
        output['py_pred'] = [output['py_pred'][-1]]
        for i in range(len(output['py_pred'])):
            py_loss += format_loss(self.py_crit, output['py_pred'][i],
                                   output['i_gt_py']) / len(output['py_pred'])
        scalar_stats.update({'py_loss': py_loss})
        loss += py_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

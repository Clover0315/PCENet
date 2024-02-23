import torch
import os
from torch import nn
import numpy as np
import torch.nn.functional
from collections import OrderedDict
from termcolor import colored

from lib.utils.snake import snake_config
from lib.utils.snake.matcher import HungarianMatcher


def sigmoid(x):
    # 通过clamp限制输出范围，防止数值溢出
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


def softmax(x):
    # 通过clamp限制输出范围，防止数值溢出
    y = torch.clamp(x.softmax(), min=1e-4, max=1 - 1e-4)
    return y


def _neg_loss(pred, gt, a=2, b=4):
    ''' cornernet 的focal loss b=4 a=2
    Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()  # 正样本索引
    neg_inds = gt.lt(1).float()  # 负样本索引

    neg_weights = torch.pow(1 - gt, b)  # 负样本权重，

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, a) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, a) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()  # 正样本数目
    pos_loss = pos_loss.sum()  # 正样本损失求和
    neg_loss = neg_loss.sum()  # 负样本损失求和

    if num_pos == 0:  # 全部都是负样本
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


def smooth_l1_loss(vertex_pred, vertex_targets, vertex_weights, sigma=1.0, normalize=True, reduce=True):
    """
    :param vertex_pred:     [b, vn*2, h, w]
    :param vertex_targets:  [b, vn*2, h, w]
    :param vertex_weights:  [b, 1, h, w]
    :param sigma:
    :param normalize:
    :param reduce:
    :return:
    """
    b, ver_dim, _, _ = vertex_pred.shape
    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets
    diff = vertex_weights * vertex_diff
    abs_diff = torch.abs(diff)
    smoothL1_sign = (abs_diff < 1. / sigma_2).detach().float()
    in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
              + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)

    if normalize:
        in_loss = torch.sum(in_loss.view(b, -1), 1) / (ver_dim * torch.sum(vertex_weights.view(b, -1), 1) + 1e-3)

    if reduce:
        in_loss = torch.mean(in_loss)

    return in_loss


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self.smooth_l1_loss = smooth_l1_loss

    def forward(self, preds, targets, weights, sigma=1.0, normalize=True, reduce=True):
        return self.smooth_l1_loss(preds, targets, weights, sigma, normalize, reduce)


class AELoss(nn.Module):
    def __init__(self):
        super(AELoss, self).__init__()

    def forward(self, ae, ind, ind_mask):
        """
        ae: [b, 1, h, w]
        ind: [b, max_objs, max_parts]
        ind_mask: [b, max_objs, max_parts]
        obj_mask: [b, max_objs]
        """
        # first index
        b, _, h, w = ae.shape
        b, max_objs, max_parts = ind.shape
        obj_mask = torch.sum(ind_mask, dim=2) != 0

        ae = ae.view(b, h * w, 1)
        seed_ind = ind.view(b, max_objs * max_parts, 1)
        tag = ae.gather(1, seed_ind).view(b, max_objs, max_parts)

        # compute the mean
        tag_mean = tag * ind_mask
        tag_mean = tag_mean.sum(2) / (ind_mask.sum(2) + 1e-4)

        # pull ae of the same object to their mean
        pull_dist = (tag - tag_mean.unsqueeze(2)).pow(2) * ind_mask
        obj_num = obj_mask.sum(dim=1).float()
        pull = (pull_dist.sum(dim=(1, 2)) / (obj_num + 1e-4)).sum()
        pull /= b

        # push away the mean of different objects
        push_dist = torch.abs(tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2))
        push_dist = 1 - push_dist
        push_dist = nn.functional.relu(push_dist, inplace=True)
        obj_mask = (obj_mask.unsqueeze(1) + obj_mask.unsqueeze(2)) == 2
        push_dist = push_dist * obj_mask.float()
        push = ((push_dist.sum(dim=(1, 2)) - obj_num) / (obj_num * (obj_num - 1) + 1e-4)).sum()
        push /= b
        return pull, push


class PolyMatchingLoss(nn.Module):
    def __init__(self, pnum):
        super(PolyMatchingLoss, self).__init__()

        self.pnum = pnum
        batch_size = 1
        pidxall = np.zeros(shape=(batch_size, pnum, pnum), dtype=np.int32)
        for b in range(batch_size):
            for i in range(pnum):
                pidx = (np.arange(pnum) + i) % pnum
                pidxall[b, i] = pidx

        device = torch.device('cuda')
        pidxall = torch.from_numpy(np.reshape(pidxall, newshape=(batch_size, -1))).to(device)

        self.feature_id = pidxall.unsqueeze_(2).long().expand(pidxall.size(0), pidxall.size(1), 2).detach()

    def forward(self, pred, gt, loss_type="L2"):
        pnum = self.pnum
        batch_size = pred.size()[0]
        feature_id = self.feature_id.expand(batch_size, self.feature_id.size(1), 2)
        device = torch.device('cuda')
        gt_expand = torch.gather(gt, 1, feature_id).view(batch_size, pnum, pnum, 2)  # 128个128个点

        pred_expand = pred.unsqueeze(1)

        dis = pred_expand - gt_expand  # pred与每一个gt的差

        if loss_type == "L2":
            dis = (dis ** 2).sum(3).sqrt().sum(2)
        elif loss_type == "L1":
            dis = torch.abs(dis).sum(3).sum(2)

        min_dis, min_id = torch.min(dis, dim=1, keepdim=True)
        # print(min_id)

        # min_id = torch.from_numpy(min_id.data.cpu().numpy()).to(device)
        # min_gt_id_to_gather = min_id.unsqueeze_(2).unsqueeze_(3).long().\
        #                         expand(min_id.size(0), min_id.size(1), gt_expand.size(2), gt_expand.size(3))
        # gt_right_order = torch.gather(gt_expand, 1, min_gt_id_to_gather).view(batch_size, pnum, 2)

        return torch.mean(min_dis)


class AttentionLoss(nn.Module):
    def __init__(self, beta=4, gamma=0.5):
        super(AttentionLoss, self).__init__()

        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, gt):
        num_pos = torch.sum(gt)
        num_neg = torch.sum(1 - gt)
        alpha = num_neg / (num_pos + num_neg)
        edge_beta = torch.pow(self.beta, torch.pow(1 - pred, self.gamma))
        bg_beta = torch.pow(self.beta, torch.pow(pred, self.gamma))

        loss = 0
        loss = loss - alpha * edge_beta * torch.log(pred) * gt
        loss = loss - (1 - alpha) * bg_beta * torch.log(1 - pred) * (1 - gt)
        return torch.mean(loss)


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class Ind2dRegL1Loss(nn.Module):
    def __init__(self, type='l1'):
        super(Ind2dRegL1Loss, self).__init__()
        if type == 'l1':
            self.loss = torch.nn.functional.l1_loss
        elif type == 'smooth_l1':
            self.loss = torch.nn.functional.smooth_l1_loss

    def forward(self, output, target, ind, ind_mask):
        """ind: [b, max_objs, max_parts]"""
        b, max_objs, max_parts = ind.shape
        ind = ind.view(b, max_objs * max_parts)
        pred = _tranpose_and_gather_feat(output, ind).view(b, max_objs, max_parts, output.size(1))
        mask = ind_mask.unsqueeze(3).expand_as(pred)
        loss = self.loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class IndL1Loss1d(nn.Module):
    def __init__(self, type='l1'):
        super(IndL1Loss1d, self).__init__()
        if type == 'l1':
            self.loss = torch.nn.functional.l1_loss
        elif type == 'smooth_l1':
            self.loss = torch.nn.functional.smooth_l1_loss

    def forward(self, output, target, ind, weight):
        """ind: [b, n]"""
        output = _tranpose_and_gather_feat(output, ind)
        weight = weight.unsqueeze(2)
        loss = self.loss(output * weight, target * weight, reduction='sum')
        loss = loss / (weight.sum() * output.size(2) + 1e-4)
        return loss


class GeoCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(GeoCrossEntropyLoss, self).__init__()

    def forward(self, output, target, poly):
        output = torch.nn.functional.softmax(output, dim=1)
        output = torch.log(torch.clamp(output, min=1e-4))
        poly = poly.view(poly.size(0), 4, poly.size(1) // 4, 2)
        target = target[..., None, None].expand(poly.size(0), poly.size(1), 1, poly.size(3))
        target_poly = torch.gather(poly, 2, target)
        sigma = (poly[:, :, 0] - poly[:, :, 1]).pow(2).sum(2, keepdim=True)
        kernel = torch.exp(-(poly - target_poly).pow(2).sum(3) / (sigma / 3))
        loss = -(output * kernel.transpose(2, 1)).sum(1).mean()
        return loss


def load_model(net, optim, scheduler, recorder, model_dir, resume=True, epoch=-1):
    if not resume:
        # os.system('rm -rf {}'.format(model_dir))
        return 0

    if not os.path.exists(model_dir):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    net.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    return pretrained_model['epoch'] + 1


def save_model(net, optim, scheduler, recorder, epoch, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(epoch)))

    # remove previous pretrained model if the number of models is too big
    # pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    # if len(pths) <= 200:
    #     return
    # os.system('rm {}'.format(os.path.join(model_dir, '{}.pth'.format(min(pths)))))


def load_network(net, model_dir, resume=True, epoch=-1, strict=True):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir) if 'pth' in pth]
    if len(pths) == 0:
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    net.load_state_dict(pretrained_model['net'], strict=strict)
    return pretrained_model['epoch'] + 1


def remove_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(prefix):
            net_[k[len(prefix):]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def add_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        net_[prefix + k] = net[k]
    return net_


def replace_net_prefix(net, orig_prefix, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(orig_prefix):
            net_[prefix + k[len(orig_prefix):]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def remove_net_layer(net, layers):
    keys = list(net.keys())
    for k in keys:
        for layer in layers:
            if k.startswith(layer):
                del net[k]
    return net


class PolyMinMatchingLoss2(nn.Module):
    # clover
    def __init__(self, pnum):
        super(PolyMinMatchingLoss2, self).__init__()

        self.pnum = pnum
        batch_size = 1
        pidxall = np.zeros(shape=(batch_size, pnum, pnum), dtype=np.int32)
        for b in range(batch_size):
            for i in range(pnum):
                pidx = (np.arange(pnum) + i) % pnum
                pidxall[b, i] = pidx

        device = torch.device('cuda')
        pidxall = torch.from_numpy(np.reshape(pidxall, newshape=(batch_size, -1))).to(device)

        self.feature_id = pidxall.unsqueeze_(2).long().expand(pidxall.size(0), pidxall.size(1), 2).detach()

    def forward(self, pred, gt, loss_type="L1"):
        pnum = self.pnum
        batch_size = pred.size(0)

        if pred.size(1) != pnum or gt.size(1) != pnum:
            return torch.tensor(0.).to(pred)
        feature_id = self.feature_id.expand(batch_size, self.feature_id.size(1), 2)
        device = torch.device('cuda')

        # pred找最近的gt
        gt_expand = torch.gather(gt, 1, feature_id).view(batch_size, pnum, pnum, 2)
        pred_expand = pred.unsqueeze(1)
        pred2gt_dis = pred_expand - gt_expand
        if loss_type == "L2":
            pred2gt_dis = (pred2gt_dis ** 2).sum(3).sqrt()
        elif loss_type == "L1":
            pred2gt_dis = torch.abs(pred2gt_dis).sum(3)
        pred2gt_min, pred2gt_min_id = torch.min(pred2gt_dis, dim=1, keepdim=True)

        # gt找最近的pred
        pred_expand = torch.gather(pred, 1, feature_id).view(batch_size, pnum, pnum, 2)
        gt_expand = gt.unsqueeze(1)
        gt2pred_dis = pred_expand - gt_expand
        if loss_type == "L2":
            gt2pred_dis = (gt2pred_dis ** 2).sum(3).sqrt()
        elif loss_type == "L1":
            gt2pred_dis = torch.abs(gt2pred_dis).sum(3)
        gt2pred_min, gt2pred_min_id = torch.min(gt2pred_dis, dim=1, keepdim=True)

        return torch.mean(torch.stack([torch.mean(pred2gt_min), torch.mean(gt2pred_min)]))


class PolyMinMatchingLoss(nn.Module):
    # clover
    def __init__(self, pnum, loss_type=2):
        super(PolyMinMatchingLoss, self).__init__()

        self.pnum = pnum
        self.loss_type = loss_type

    def forward(self, pred, gt):
        pnum = self.pnum
        if pred.size(1) != pnum or gt.size(1) != pnum:
            return torch.tensor(0.).to(pred)

        loss_type = self.loss_type
        matrix = torch.cdist(pred, gt, p=loss_type)
        # pred找最近的gt
        pred2gt_min, pred2gt_min_id = torch.min(matrix, dim=1, keepdim=True)
        # gt找最近的pred
        gt2pred_min, gt2pred_min_id = torch.min(matrix, dim=2, keepdim=True)

        return torch.mean(torch.stack([torch.mean(pred2gt_min), torch.mean(gt2pred_min)]))


class DMLLoss(nn.Module):
    # BuildMapper
    def __init__(self, pnum):
        super(DMLLoss, self).__init__()

        self.pnum = pnum

    def forward(self, pred, gt, gt_valid, loss_type=1):
        pnum = self.pnum
        if pred.size(1) != pnum or gt.size(1) != pnum:
            return torch.tensor(0.).to(pred)

        matrix = torch.cdist(pred, gt, p=loss_type)
        # pred找最近的gt
        pred2gt_min, pred2gt_min_id = torch.min(matrix, dim=1, keepdim=True)
        # gt_valid找最近的pred
        matrix = torch.cdist(gt_valid, pred, p=loss_type)
        gt2pred_min, gt2pred_min_id = torch.min(matrix, dim=1, keepdim=True)

        return torch.mean(torch.stack([torch.mean(pred2gt_min), torch.mean(gt2pred_min)]))


class HungarianLoss(nn.Module):
    # clover
    def __init__(self):
        super(HungarianLoss, self).__init__()
        matcher = HungarianMatcher()
        self.matcher = matcher

    def _get_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _, _) in indices])
        tgt_idx = torch.cat([tgt for (_, tgt, _) in indices])
        h_loss = torch.stack([h_loss for (_, _, h_loss) in indices])
        return batch_idx, src_idx, tgt_idx, h_loss

    def forward(self, outputs, targets, l_type=2, loss_type='balance_exp'):
        device = outputs['pred_poly'].device
        if not outputs['pred_poly'].size(0) or not targets['gt_py'].size(0):
            return torch.tensor(0.).to(device)
        pnum = outputs['pred_poly'].size(1)
        # 匈牙利匹配
        indices = self.matcher(outputs, targets, l_type)
        b_idx, src_idx, tgt_idx, h_loss = self._get_permutation_idx(indices)  # batch_idx, src_idx
        sizes = [int(v) for v in targets['gt_num']]
        pred_prob = outputs['pred_prob']

        # 分类矩阵
        gt = torch.zeros_like(pred_prob).to(device)
        gt[b_idx, src_idx] = 1
        # 交叉熵损失
        if loss_type == 'cross':
            pred_prob[b_idx, src_idx] = 1 - pred_prob[b_idx, src_idx]
            p_loss = -torch.log(1 - pred_prob)
            p_loss = torch.mean(p_loss)
        # 固定权重交叉熵损失
        elif loss_type == 'w_cross':
            pred_prob[b_idx, src_idx] = 1. - pred_prob[b_idx, src_idx]
            p_loss = -torch.log(1. - pred_prob)
            m = torch.where(gt == 1, 1., 0.1)  #
            p_loss = m * p_loss
            p_loss = torch.mean(torch.sum(p_loss, 1))
        # 平衡交叉熵损失
        elif loss_type == 'balance':
            pred_prob[b_idx, src_idx] = 1 - pred_prob[b_idx, src_idx]
            m = torch.tensor(sizes).repeat(pnum, 1).permute(1, 0).to(device)
            m = torch.where(gt == 1, pnum / m, pnum / (pnum - m))
            p_loss = m * -torch.log(1 - pred_prob)
            p_loss = torch.mean(p_loss)
        # 平衡指数损失
        elif loss_type == 'balance_exp':
            min_d = 0.05
            # 权重矩阵m
            m = torch.tensor(sizes).repeat(pnum, 1).permute(1, 0).to(device)
            m = torch.where(gt == 1, pnum / m, pnum / (pnum - m))
            pred_prob[b_idx, src_idx] = torch.clamp(pred_prob[b_idx, src_idx], min=min_d)
            p_loss = m * torch.where(gt == 1, 1. / pred_prob - 1., -torch.log(1 - pred_prob))  # 正样本指数，负样本对数
            # a = 1.
            # pred_prob = 1. - pred_prob
            # pred_prob[b_idx, src_idx] = 1. - pred_prob[b_idx, src_idx]
            # # 避免求倒数时内存溢出，限制最小值
            # pred_prob = torch.clamp(pred_prob, min=min_d)
            # p_loss = m * torch.pow((1. / pred_prob - 1.), a)

            p_loss = torch.mean(p_loss)
        # focal loss
        elif loss_type == 'focal':
            r = 2
            b = torch.where(gt == 1, 1 - pred_prob, pred_prob)
            pred_prob[b_idx, src_idx] = 1 - pred_prob[b_idx, src_idx]
            # 权重矩阵
            m = torch.tensor(sizes).repeat(pnum, 1).permute(1, 0).to(device)
            m = torch.where(gt == 1, pnum / m, pnum / (pnum - m))
            # a = torch.where(gt == 1, 0.25, 0.75)  # 固定样本平衡系数
            # a = torch.where(gt == 1, 0.9, 0.1)  # 固定样本平衡系数
            p_loss = - m * torch.pow(b, r) * torch.log(1 - pred_prob)
            # p_loss = torch.mean(p_loss.sum(1))

            # p_loss = - torch.pow(b, r) * torch.log(1 - pred_prob)
            p_loss = torch.mean(p_loss)
        else:
            raise ValueError('loss_type 不合法')

        h_w = 0.1  # h_loss乘以h_w，以平衡度量衡
        h_loss = torch.mean(h_loss) * h_w
        return p_loss + h_loss


if __name__ == "__main__":
    polymathchloss = PolyMinMatchingLoss(4)
    gt = torch.Tensor(
        [
            [
                [0, 0],
                [0, 1],
                [1, 1],
                [1, 0]
            ]
        ]
    ).to('cuda')
    pred = torch.Tensor(
        [
            [
                [0, 0],
                [-2, 2],
                [1, 1],
                [1, 0]
            ]
        ]
    ).to('cuda')
    loss = polymathchloss(pred, gt, 1)
    print(loss)

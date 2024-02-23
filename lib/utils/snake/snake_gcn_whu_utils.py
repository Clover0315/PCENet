import random

import torch
from lib.utils.snake import snake_decode, snake_config, snake_gcn_utils, snake_whu_utils
from lib.csrc.extreme_utils import _ext as extreme_utils
import cv2
import numpy as np

def translate_poly_batch(img_poly, offset, h, w):
    '''
    多边形平移

    :param img_poly:
    :param offset:
    :param h:
    :param w:
    :return:
    '''
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)

    # clone img_poly to can_poly
    can_poly = img_poly.clone()
    can_poly = can_poly + offset

    # create a new tensor to store the result of clamping
    clamped_can_poly = torch.clamp(can_poly, min=0, max=w - 1)
    clamped_can_poly = torch.clamp(clamped_can_poly, min=0, max=h - 1)

    return clamped_can_poly


def translate_poly(img_poly, offset, h, w):
    '''
    多边形平移

    :param img_poly:
    :param offset:
    :param h:
    :param w:
    :return:
    '''
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)

    # clone img_poly to can_poly
    can_poly = img_poly.clone()
    can_poly[..., 0] = can_poly[..., 0] + offset[0]
    can_poly[..., 1] = can_poly[..., 1] + offset[1]

    # create a new tensor to store the result of clamping
    clamped_can_poly = torch.clamp(can_poly, min=0, max=w - 1)
    clamped_can_poly = torch.clamp(clamped_can_poly, min=0, max=h - 1)

    return clamped_can_poly


def can_poly(pys):
    if not snake_config.pan_center:
        return snake_gcn_utils.img_poly_to_can_poly(pys)  # 平移平移到边缘（左下角）
    else:
        return img_poly_to_center_poly(pys)  # 平移平移到边缘（左下角）


def get_init(box):
    x_min, y_min, x_max, y_max = box[..., 0], box[..., 1], box[..., 2], box[..., 3]

    w = x_max - x_min
    h = y_max - y_min
    box = [
        x_min, y_min,
        x_min, y_max,
        x_max, y_max,
        x_max, y_min,


        # x_min, (y_min + y_max) / 2.,
        # (x_min + x_max) / 2., y_max,
        # x_max, (y_min + y_max) / 2.,
        # (x_min + x_max) / 2., y_min,


        # x_min, y_min + 0.125 * h,
        # x_min, y_min + 0.875 * h,
        # x_min + 0.125 * w, y_max,
        # x_min + 0.875 * w, y_max,
        # x_max, y_min + 0.875 * h,
        # x_max, y_min + 0.125 * h,
        # x_min + 0.875 * w, y_min,
        # x_min + 0.125 * w, y_min,
    ]
    # box = torch.stack(box, dim=2).view(x_min.size(0), x_min.size(1), 4, 2)
    box = torch.stack(box, dim=2).view(x_min.size(0), x_min.size(1), 4, 2)
    return box


def prepare_testing_init(box, score):
    '''
    从bbox中解析出初始化轮廓
    :param box:
    :param score:
    :return:i_it_4py Tensor(轮廓数目，40， 2)[初始化菱形]
    c_it_4py
    ind
    '''
    i_it_4pys = get_init(box)
    i_it_4pys = snake_gcn_utils.uniform_upsample(i_it_4pys, snake_config.init_poly_num)  # 每条边上均匀采样
    c_it_4pys = can_poly(i_it_4pys)  # 平移平移到边缘（左下角）

    # 筛选分数大于阈值的轮廓
    ind = score > snake_config.ct_score
    i_it_4pys = i_it_4pys[ind]
    c_it_4pys = c_it_4pys[ind]

    ind = torch.cat([torch.full([ind[i].sum()], i) for i in range(ind.size(0))], dim=0)
    init = {
        'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys,
        'ind': ind
    }

    return init


def prepare_testing_evolve(ex):
    if len(ex) == 0:  # 没有极点
        i_it_pys = torch.zeros([0, snake_config.poly_num, 2]).to(ex)
        c_it_pys = torch.zeros_like(i_it_pys)
    else:
        i_it_pys = snake_gcn_utils.uniform_upsample(ex[None], snake_config.poly_num)[0]  # 均匀采样

        c_it_pys = can_poly(i_it_pys)  # 八边形平移到边缘
    evolve = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys}
    return evolve


def uniform_upsample(poly, p_num):
    '''
    非均匀采样，每条边上采样数目相等
    :param poly:
    :param p_num:
    :return:
    '''
    # 1. assign point number for each edge
    # 2. calculate the coefficient for linear interpolation  计算线性插值的系数
    b_num = poly.size(1)  # batch num
    if not b_num:
        return torch.zeros(size=(poly.size(0), poly.size(1), p_num, poly.size(3)))
    v_num = poly.size(2)  # 原来的端点数

    next_poly = torch.roll(poly, -1, 2)  # 另一个端点
    edge_len = (next_poly - poly).pow(2).sum(3).sqrt()  # 边长
    edgeidxsort_p = torch.argsort(edge_len, dim=2)

    edge_num = torch.LongTensor([p_num//v_num]*(v_num*b_num)).view(1, b_num, v_num).to(poly.device)  # 每条边上采样端点数
    diff_num = p_num % v_num  # 还差的端点数
    if diff_num:
        edge_id = edgeidxsort_p[..., v_num-diff_num:]
        diff_indx = torch.zeros((1, b_num, v_num), dtype=torch.long).to(poly.device)
        diff_indx.scatter_(2, edge_id, 1)
        edge_num += diff_indx

    edge_num_sum = torch.sum(edge_num, dim=2)
    assert torch.all(edge_num_sum == p_num)  #

    edge_start_idx = torch.cumsum(edge_num, dim=2) - edge_num
    weight, ind = extreme_utils.calculate_wnp(edge_num, edge_start_idx, p_num)
    poly1 = poly.gather(2, ind[..., 0:1].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly2 = poly.gather(2, ind[..., 1:2].expand(ind.size(0), ind.size(1), ind.size(2), 2))
    poly = poly1 * (1 - weight) + poly2 * weight

    return poly


def img_poly_to_center_poly(img_poly):
    '''
    平移多边形，使得中心点在原点,即向左平移(x_min + x_max)*0.5，向下平移(y_min + y_max)*0.5
    :param img_poly:
    :return:
    '''
    if len(img_poly) == 0 or img_poly.size(1) == 0:
        return torch.zeros_like(img_poly)
    x_min = torch.min(img_poly[..., 0], dim=-1)[0]
    y_min = torch.min(img_poly[..., 1], dim=-1)[0]
    x_max = torch.max(img_poly[..., 0], dim=-1)[0]
    y_max = torch.max(img_poly[..., 1], dim=-1)[0]
    diff_x = x_max - x_min
    diff_y = y_max - y_min
    ct_x = (x_min + x_max)*0.5  # 中点x
    ct_y = (y_min + y_max)*0.5  # 中点y
    can_poly = img_poly.clone()
    can_poly[..., 0] = (can_poly[..., 0] - ct_x[..., None]) / diff_x[..., None]
    can_poly[..., 1] = (can_poly[..., 1] - ct_y[..., None]) / diff_y[..., None]
    return can_poly


def get_poly_diagonal(poly):
    '''
    获取poly的对角线
    :param poly: batch. poly_num, 2
    :return:
    '''
    mmin = torch.min(poly, dim=1)[0]
    min_x, min_y = mmin[..., 0], mmin[..., 1]  # batch 1
    mmax = torch.max(poly, dim=1)[0]
    max_x, max_y = mmax[..., 0], mmax[..., 1]
    lt = torch.stack((min_x, min_y), dim=1)  # 左上 batch 2
    rb = torch.stack((max_x, max_y), dim=1)  # 右下
    l1 = torch.stack((lt, rb), dim=1)
    l1 = snake_gcn_utils.uniform_upsample(l1[None], 64)[0]

    rt = torch.stack((max_x, min_y), dim=1)
    lb = torch.stack((min_x, max_y), dim=1)
    l2 = torch.stack((rt, lb), dim=1)  # 右对角线
    l2 = snake_gcn_utils.uniform_upsample(l2[None], 64)[0]

    line = torch.cat((l1, l2), dim=1)
    return line

# ---------------------------------------------------#
#   设置种子
# ---------------------------------------------------#
def seed_everything(seed=3157):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 设置torch的随机种子
    torch.cuda.manual_seed(seed)  # 设置GPU的随机种子
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True  # 开启确定性算法，会使得训练变慢
    # torch.backends.cudnn.benchmark = False  # 自动寻找最适合当前配置的高效算法，如果输入每个iteration的数据不一样，设置为True会降低训练速度


def prepare_training_ep_tp(ret, batch):
    ct_01 = batch['ct_01'].byte()
    init = {}

    init.update({'gt_pys': snake_gcn_utils.collect_training(batch['gt_pys'], ct_01)})
    init.update({'gt_pys_num': snake_gcn_utils.collect_training(batch['gt_pys_num'], ct_01)})

    ct_num = batch['meta']['ct_num']
    init.update({'py_ind': torch.cat([torch.full([ct_num[i]], i) for i in range(ct_01.size(0))], dim=0)})

    init['py_ind'] = init['py_ind'].to(ct_01.device)

    return init

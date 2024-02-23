import cv2
import numpy as np

from lib.utils.snake import snake_coco_utils
from munkres import Munkres
from scipy import optimize


def get_extreme_points2(pts):
    '''
    最小外接矩形与多边形的交点
    :param pts: 多边形 ndarray(N, 2)
    :return: 极点数组，极点数目为4~8个
    '''
    # cv只能处理float32，这里转一下数据类型
    pts_32 = pts.astype(np.float32)
    rect = cv2.minAreaRect(pts_32)  # 最小外接矩形，x,y,l,w
    # cv2.boxPoints可以将轮廓点转换为四个角点坐标
    # 最小外接矩形
    box = cv2.boxPoints(rect)
    extreme_ids = []
    thresh = 0.5
    for i, point in enumerate(pts):
        dis = cv2.pointPolygonTest(box, tuple(point.astype(np.float32)), True)
        if dis <= thresh:
            extreme_ids.append(i)

    extreme_point = pts[extreme_ids]
    return extreme_point, box


def get_min_area_rect(pts):
    '''
    获取外接矩形
    :param pts: 多边形 ndarray(N, 2)
    '''
    # minAreaRect使用int32 int64 float32
    pts = pts.astype(np.int64)
    rect = cv2.minAreaRect(pts)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    box = cv2.boxPoints(rect)  # 可以将轮廓点转换为四个角点坐标
    return box


def can_poly_lb(poly):
    '''
    将多边形起点移动到minx miny
    :param poly:
    :return:
    '''
    x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
    tt_idx = np.argmin(poly[:, 0] - x_min)
    poly = np.roll(poly, -tt_idx, axis=0)
    tt_poly = poly[np.where(poly[:, 0] == x_min)]
    tt_idx = np.argmin(tt_poly[:, 1] - y_min)
    poly = np.roll(poly, -tt_idx, axis=0)
    return poly


def get_min_extreme_points(pts, min_rect):
    '''
    获取最小外接矩形（最小外接矩阵与多边形的交点）
    :param pts: 多边形轮廓
    :param min_rect: 最小外接矩形
    :return:
    '''
    result = []
    thresh = 1  # 阈值
    for point in pts:
        dis = cv2.pointPolygonTest(min_rect, tuple(point), True)
        if abs(dis) <= thresh:
            result.append(point)
    if len(result) == 0:  # 没有匹配
        return min_rect
    return np.asarray(result)


def get_extreme_points(pts):
    '''
    获取极点
    :param pts: 多边形 ndarray(N, 2)
    :return: 极点数组，极点数目为4~8个
    '''
    l, t = min(pts[:, 0]), min(pts[:, 1])  # left top right bottom
    r, b = max(pts[:, 0]), max(pts[:, 1])
    extreme_val = [l, t, r, b]
    # 3 degrees
    thresh = 0.01
    w = r - l + 1
    h = b - t + 1

    # vertex_num = pts.shape[0]
    l_idx = np.argmin(pts[:, 0])
    t_idx = np.argmin(pts[:, 1])  # t的索引
    r_idx = np.argmax(pts[:, 0])
    b_idx = np.argmax(pts[:, 1])

    idxs = [l_idx, t_idx, r_idx, b_idx]
    flags = [0, 1, 0, 1]  # 是x还是y
    extreme_ids = []
    '''与t的y差值在阈值以内，取max x 、min x为极点'''
    for (_val, _idx, flag) in zip(extreme_val, idxs, flags):
        w_or_h = h if flag else w
        ex_idxs = np.where(abs(pts[:, flag] - _val) <= thresh * w_or_h)[0]  # 极值点的索引
        ex_points = pts[ex_idxs]  # 极值点的值
        # 极值点连线上的两个端点的索引
        _flag = 0 if flag else 1
        min_idx = np.argmin(ex_points[:, _flag])
        extreme_ids.append(ex_idxs[min_idx])
        max_idx = np.argmax(ex_points[:, _flag])
        extreme_ids.append(ex_idxs[max_idx])
    extreme_ids.sort()
    extreme_ids = np.unique(extreme_ids)  # 删除重复的点
    extreme_point = pts[extreme_ids]
    return extreme_point


def get_init(box):
    x_min, y_min, x_max, y_max = box

    # w = x_max - x_min
    # h = y_max - y_min
    box = [
        # bbox
        [x_min, y_min],
        [x_min, y_max],
        [x_max, y_max],
        [x_max, y_min]

        # bbox + 中点
        # [x_min, (y_min + y_max) / 2.],
        # [(x_min + x_max) / 2., y_max],
        # [x_max, (y_min + y_max) / 2.],
        # [(x_min + x_max) / 2., y_min],

        # 八边形
        # [x_min, y_min + 0.125 * h],
        # [x_min, y_min + 0.875 * h],
        # [x_min + 0.125 * w, y_max],
        # [x_min + 0.875 * w, y_max],
        # [x_max, y_min + 0.875 * h],
        # [x_max, y_min + 0.125 * h],
        # [x_min + 0.875 * w, y_min],
        # [x_min + 0.125 * w, y_min],
    ]

    return np.array(box)


def uniformsample(pgtnp_px2, newpnum):
    '''
    多边形重采样，每条边的采样数相近
    :param pgtnp_px2: 原始多边形数组
    :param newpnum: 目标端点数
    :return:
    '''
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum  # 每个端点的下一个端点索引
    pgtnext_px2 = pgtnp_px2[idxnext_p]  # 每个端点下一个端点的坐标
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))  # 每条边的长度
    edgeidxsort_p = np.argsort(edgelen_p)  # 边长从小到大排序的索引

    # two cases
    # we need to remove gt points
    # # 目标端点数少于原始端点数时，从原始的端点中删除一部分（删除较短的路径）
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]  # 删除较短的路径的索引
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.array([newpnum // pnum] * pnum, dtype=np.int32)  # 每条边上分配的端点数相等
        diffnum = newpnum % pnum  # 还缺少的端点数
        if diffnum:  # 在较长的diffnum条边上各增加1一个端点
            edgeid = edgeidxsort_p[pnum - diffnum:]
            edgenum[edgeid] += 1

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]  # 起始端点
            pe_1x2 = pgtnext_px2[i:i + 1]  # 结尾端点

            pnewnum = edgenum[i]  # 此边上要生成的端点数
            wnp_kx1 = np.arange(pnewnum, dtype=np.float32).reshape(-1, 1) / pnewnum  # △数组

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1  # 计算每条边上的端点坐标
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp


def img_poly_to_center_poly(img_poly, x_min, y_min, x_max, y_max):
    '''
    平移多边形，使得中心点在原点,即向左平移(x_min + x_max)*0.5，向下平移(y_min + y_max)*0.5
    :param img_poly:
    :param x_min:
    :param y_min:
    :param x_max:
    :param y_max:
    :return:
    '''
    ct_x = (x_min + x_max) * 0.5
    ct_y = (y_min + y_max) * 0.5
    diff_x = x_max - x_min
    diff_y = y_max - y_min
    can_poly = (img_poly - np.array([ct_x, ct_y])) / np.array([diff_x, diff_y])
    return can_poly


def uniformsample_aug(pgtnp_px2, newpnum, return_v=False):
    '''
    多边形重采样，按边长均匀分配每条边的采样数
    :param pgtnp_px2: 原始多边形
    :param newpnum: 目标端点数
    :return:
    '''
    pnum, cnum = pgtnp_px2.shape  # 边数/点数
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum  # 每个端点的下一个端点索引
    pgtnext_px2 = pgtnp_px2[idxnext_p]  # 每个端点下一个端点的坐标
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))  # 每条边的长度
    edgeidxsort_p = np.argsort(edgelen_p)  # 边长从小到大排序的索引

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum >= newpnum:  # 目标端点数少于原始端点数时，从原始的端点中删除一部分
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum

        if return_v:
            return pgtnp_kx2, np.array([1] * newpnum, dtype=np.int64)  # 全是端点
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)  # 按比例分配每条边上的点数
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # round操作后总点数可能与目标数有差异，在这里处理
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:  # 总点数多于目标数，要去除多于的点数
                    edgeid = edgeidxsort_p[id]  # 从最长的一条开始减去多出的节点数
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:  # 总点数少于目标数，在最长的一条边上增加缺少的点数
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        endpoint_type = []  # 记录端点的类型，1为端点，0为生成的（非端点）

        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]  # 起始端点
            pe_1x2 = pgtnext_px2[i:i + 1]  # 结尾端点

            pnewnum = edgenum[i]  # 此边上要生成的端点数
            wnp_kx1 = np.arange(pnewnum, dtype=np.float32).reshape(-1, 1) / pnewnum

            # 生成数据增强随机变量
            aug_param = np.random.normal(0, 0.2, (pnewnum, 2))
            # aug_param = np.random.uniform(-0.4, 0.4, (pnewnum, 2))
            # 随机将一些数值设置为0
            # aug_param *= np.random.choice([0, ] * (9-pnum) + [1, ] * (pnum-3), (pnewnum, 2))
            # aug_param *= np.random.choice([0, 1], (pnewnum, 2))
            aug_param = np.sort(aug_param, axis=0)  # 按大小排序
            rolled_arr = []
            for _i in range(2):
                even_nd = aug_param[0::2, _i]
                odd_nd = aug_param[1::2, _i][::-1]
                ndr = np.concatenate((even_nd, odd_nd))
                # 随机滚动
                t = np.random.randint(0, pnewnum)
                rolled_arr.append(np.roll(ndr, t))
                # t1 = np.random.randint(0, pnewnum+1)
                # t2 = np.random.randint(t1, pnewnum+1)

                # ndr = ndr[t1:t2]
                # rolled_arr.append(np.concatenate((np.zeros(t1),ndr, np.zeros((pnewnum-t2)))))
            aug_param = np.stack(rolled_arr, axis=1)

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1 + aug_param  # 计算每条边上的端点坐标
            psample.append(pmids)
            endpoint_type.append([1] + [0] * (pnewnum - 1))

        psamplenp = np.concatenate(psample, axis=0)
        endpoint_types = np.concatenate(endpoint_type, axis=0)
        if return_v:
            return psamplenp, endpoint_types
        return psamplenp


def get_approx_poly(ndr, num):
    # 多边形拟合
    approx_poly = []
    for poly in ndr:
        poly = cv2.approxPolyDP(poly.cpu().numpy(), 0.5, False)  # 多边形拟合
        poly = np.squeeze(poly)
        # 将poly的第一个点与(x_min, y_min) 对齐
        x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
        tt_idx = np.argmin(poly[:, 0] - x_min)
        poly = np.roll(poly, -tt_idx, axis=0)
        tt_poly = poly[np.where(poly[:, 0] == x_min)]
        tt_idx = np.argmin(tt_poly[:, 1] - y_min)
        poly = np.roll(poly, -tt_idx, axis=0)

        poly = snake_coco_utils.uniformsample(poly, num)
        approx_poly.append(poly)
    approx_poly = np.array(approx_poly)
    return approx_poly


def cal_munkres(py_pred, py_gt, matrix_type='L2'):
    '''
    计算最优匹配
    :param py_pred:
    :param py_gt:
    :return:
    '''
    py_pred = np.expand_dims(py_pred, axis=1)
    py_gt = np.tile(py_gt, (py_pred.shape[0], 1, 1))

    pred2gt = py_pred - py_gt
    if matrix_type == 'L2':
        matrix = np.sqrt((pred2gt ** 2).sum(2))  # l2距离，欧式距离
    elif matrix_type == 'L1':
        matrix = np.abs(pred2gt).sum(2)  # l1距离，曼哈顿距离

    # m = Munkres()
    # indexes_cost = m.compute(matrix.tolist())  # 求最小消耗
    #
    # return np.array(indexes_cost, dtype=np.uint32)

    origin_id, target_id = optimize.linear_sum_assignment(matrix)
    return origin_id

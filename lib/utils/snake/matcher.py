# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_dis: float = 5.):
        """
        Params:
            cost_class: 分类损失参数
            cost_dis: 检测框间L1距离损失参数
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_dis = cost_dis
        assert cost_class != 0 or cost_dis != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, l_type=2):
        """ 匈牙利匹配
        Params:
            outputs:
                 "pred_prob": 推理分数:batch * 点数
                 "pred_py": 推理多边形：batch * 点推理数 * 2

            targets:
                 "gt_num": 目标多边形的真实点数
                 "gt_py": 真实多边形坐标，不足poly_num时补零

        Returns:
            [( 批1(index_i, index_j)，批2……]
                - index_i output中被匹配的索引
                - index_j target中的索引
                len(index_i) = len(index_j) = min(poly_num, num_target_boxes)
        """
        out_poly = outputs["pred_poly"]  # batch_size * poly_num, 2
        tgt_num = targets["gt_num"]  # batch_size
        tgt_poly = targets["gt_py"]  # batch_size * poly_num, 2

        cost_dis = torch.cdist(out_poly, tgt_poly, p=l_type)  # poly_num*poly_num, 距离
        C = cost_dis
        C = C.cpu()

        # 匈牙利匹配
        indices = []
        i = 0
        for c in C:
            if tgt_num.size(0):  # gt为空的情况
                if int(tgt_num[i]):
                    c = c[..., :int(tgt_num[i])]  # 按 tgt_num 将matrix截取成128*tgt_num
            idx = linear_sum_assignment(c)
            h_cost = c[idx].sum()  # 匈牙利匹配得分（越低越好, 0表示完全重合）
            indices.append([idx, h_cost])
            i += 1

        return [(
            torch.as_tensor(out_ids, dtype=torch.int64),
            torch.as_tensor(tgt_ids, dtype=torch.int64),
            torch.as_tensor(h_cost, dtype=torch.float32),
        ) for [(out_ids, tgt_ids), h_cost] in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_dis=args.set_cost_bbox)


def test_hungarian():
    p_num = 128
    pred_prob = torch.rand(1, p_num).cuda()
    pred_poly = torch.rand(1, p_num, 2).cuda()  # 推理多边形：batch * 点推理数 * 2

    gt_num = torch.randint(high=20, low=4, size=(1,)).cuda()
    gt_py = torch.cat([torch.rand(int(gt_num[0]), 2), torch.zeros((p_num - int(gt_num[0]), 2))]).unsqueeze(0).cuda()

    matcher = HungarianMatcher()
    indices_single = matcher({'pred_prob': pred_prob, 'pred_poly': pred_poly},
                             {'gt_num': gt_num, 'gt_py': gt_py})
    indices_batched = matcher({'pred_prob': pred_prob.repeat(2, 1),
                               'pred_poly': pred_poly.repeat(2, 1, 1)},
                              {
                                  'gt_num': gt_num.repeat(2),
                                  'gt_py': gt_py.repeat(2, 1, 1)}
                              )

    # test with empty targets
    gt_num_empty = torch.randint(high=20, low=4, size=(0,))
    gt_py_empty = torch.rand(0, 2)
    # indices = matcher({'pred_prob': pred_prob.repeat(2, 1),
    #                    'pred_poly': pred_poly.repeat(2, 1, 1)}, {
    #                       'gt_num': torch.cat([gt_num, gt_num_empty]),
    #                       'gt_py': torch.cat([gt_py, gt_py_empty])})

    # indices = matcher({'pred_prob': pred_prob.repeat(2, 1),
    #                    'pred_poly': pred_poly.repeat(2, 1, 1)}, {
    #                       'gt_num': gt_num_empty.repeat(2),
    #                       'gt_py': gt_py_empty.repeat(2, 1, 1)})

    pass


if __name__ == "__main__":
    test_hungarian()

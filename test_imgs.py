import math

import cv2

from lib import set_task_name

set_task_name('ep_tp')
import warnings

warnings.filterwarnings('ignore')

from lib.config import cfg
from lib.networks import make_network
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network

import torch
import tqdm
import matplotlib.pyplot as plt
from lib.utils.snake import snake_whu_utils, snake_config
from lib.utils import img_utils
import numpy as np
from itertools import cycle
from matplotlib.patches import Ellipse, Circle

mean = snake_config.mean
std = snake_config.std


def visual_datalist(batch, ex, ratio=True, scatter=False):
    inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))

    ex = ex[-1] if isinstance(ex, list) else ex
    ex = ex.detach().cpu().numpy() if isinstance(ex, torch.Tensor) else ex
    ex = ex * snake_config.down_ratio if ratio else ex

    fig, ax = plt.subplots(1, figsize=[10, 10])
    fig.tight_layout()
    ax.axis('off')
    ax.imshow(inp)

    colors = np.array([
        # [0, 0, 255],
        # [0, 206, 209],
        # [255, 255, 0],
        # [178, 34, 34],
        [255, 0, 0],
        # [255, 20, 147],
        # [0, 238, 238],
        # [0, 255, 127],
        # [0, 255, 0]
    ]) / 255.
    colors = cycle(colors)

    # ex = ex[1:3]
    for i in range(len(ex)):
        poly = ex[i]
        poly = poly.detach().cpu().numpy() if isinstance(poly, torch.Tensor) else poly
        color = next(colors).tolist()
        poly = np.append(poly, [poly[0]], axis=0)
        ax.plot(poly[:, 0], poly[:, 1], linewidth=1, color=color, zorder=1)
        # ax.fill(poly[:, 0], poly[:, 1], alpha=0.3, color=color, zorder=0)
        if scatter:
            ax.scatter(poly[:, 0], poly[:, 1], color=color, s=10, zorder=2, edgecolor='w')

    plt.show()


def visual_curve(batch, boxes):
    inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))

    fig, ax = plt.subplots(1, figsize=[10, 10])
    fig.tight_layout()
    ax.axis('off')
    ax.imshow(inp)

    colors = np.array([
        [0, 0, 255],
        [0, 206, 209],
        [255, 255, 0],
        [178, 34, 34],
        [255, 0, 0],
        [255, 20, 147],
        [0, 238, 238],
        [0, 255, 127],
        [0, 255, 0]
    ]) / 255.
    colors = cycle(colors)

    # ex = ex[1:3]
    boxes = boxes.detach().cpu().numpy() * snake_config.down_ratio
    for i in range(len(boxes)):
        box = boxes[i]
        min_x, min_y = min(box[..., 0]), min(box[..., 1])
        max_x, max_y = max(box[..., 0]), max(box[..., 1])
        xy = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5)
        color = next(colors).tolist()
        width = max_x - min_x
        height = max_y - min_y
        e = Ellipse(xy=xy, width=width, height=height, angle=0,
                    edgecolor=color, linewidth=3, facecolor='none')
        ax.add_patch(e)

    plt.show()


def visual_single_data(batch, ex):
    inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))

    scale = 4
    o_size = (inp.size(0) * scale, inp.size(1) * scale)
    inp = cv2.resize(inp.detach().cpu().numpy(), o_size)
    ex = ex[-1] if isinstance(ex, list) else ex
    ex = ex.detach().cpu().numpy() if isinstance(ex, torch.Tensor) else ex
    ex = ex * snake_config.down_ratio * scale
    colors = np.array([
        # [0, 0, 255],
        # [0, 206, 209],
        # [255, 255, 0],
        # [178, 34, 34],
        [255, 0, 0],
        # [255, 20, 147],
        # [0, 238, 238],
        # [0, 255, 127],
        # [0, 255, 0]
    ]) / 255.
    colors = cycle(colors)

    for poly in [ex[7],ex[9],ex[12],ex[15]]:
        fig, ax = plt.subplots(1, figsize=[10, 10])
        fig.tight_layout()
        ax.axis('off')

        x_min, x_max = math.floor((min(poly[..., 0]))), math.ceil(max(poly[..., 0]))
        y_min, y_max = math.floor((min(poly[..., 1]))), math.ceil(max(poly[..., 1]))
        ax.imshow(inp[y_min:y_max, x_min:x_max, :])  # 裁切
        # poly = ex[i]
        poly = poly.detach().cpu().numpy() if isinstance(poly, torch.Tensor) else poly
        poly = poly - (x_min, y_min)
        color = next(colors).tolist()
        poly = np.append(poly, [poly[0]], axis=0)
        ax.plot(poly[:, 0], poly[:, 1], linewidth=4, color=color, zorder=1)
        plt.show()


def val(network, data_loader, visual_map):
    network.eval()
    torch.cuda.empty_cache()

    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()

        with torch.no_grad():
            output = network(batch['inp'], batch)

        '''km 算法匹配py和'''
        py = output['py']
        py = py[-1] if isinstance(py, list) else py
        py = py.detach().cpu().numpy()

        gt = batch['meta']['gt_py']

        pys_valid = []
        for i in range(len(gt)):
            py_pred = py[i]
            py_gt = gt[i].detach().cpu().numpy()[0]
            indexes_cost = snake_whu_utils.cal_munkres(py_pred, py_gt)
            py_valid = py_pred[indexes_cost, :]
            pys_valid.append(py_valid)

        '''curve'''
        # visual_curve(batch, output['i_it_4py'])
        for name in visual_map:
            if 'ex' == name:
                visual_datalist(batch, output['ex'])
            if 'py' == name:
                visual_datalist(batch, output['py'], True, False)
                # visual_single_data(batch, output['py'])
            if 'valid' == name:
                visual_datalist(batch, np.array(pys_valid), True)
            if 'gt' == name:
                visual_datalist(batch, output['i_gt_py'], False)
            if 'egt' == name:
                visual_datalist(batch, output['i_gt_4py'])
            if 'snake_ex' == name:
                visual_datalist(batch, output['i_it_py'])
            if 'box' == name:
                visual_datalist(batch, output['i_it_4py'])

            if 'valid_py' == name:
                visual_datalist(batch, [[_ * snake_config.down_ratio for _ in output['valid_py']]], True, True)


def main():
    cfg.task = 'snake'
    # cfg.task = 'ep_tp'
    cfg.test_km = True  # 测试km分配

    img_ids = [
        # 134,
        # 800000105040,  # *
        # 800000302935,  # *
        # 800000200495,
        800000103958,  # *
        # 800000201398, 800000200660,
        # 15120000002676, 15120000002708, 15120000003735,
        # 158973, 28142, 34888
    ]

    network1 = make_network(cfg).cuda()
    # 加载模型
    # load_network(network1, 'data/model/34/1/snake_ep_tp/whu/')
    load_network(network1, 'data/model/2/snake/whu')
    # load_network(network1, 'data/model/4/snake3/whu')
    # load_network(network1, 'data/model/22/snake_ep_tp/whu')
    # 加载数据集
    # val_loader1 = make_data_loader(cfg, is_train=False)
    val_loader1 = make_data_loader(cfg, is_train=False, img_ids=img_ids)
    val(network1, val_loader1, ['valid'])
    #
    # cfg.task = 'snake'
    # network2 = make_network(cfg).cuda()
    # # 加载模型
    # load_network(network2, 'data/model/2/snake/whu')
    # # 加载数据集
    # val_loader2 = make_data_loader(cfg, is_train=False, img_ids=img_ids)
    # val(network2, val_loader2, ['py', 'valid', 'gt'])


if __name__ == "__main__":
    main()

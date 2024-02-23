from lib import set_task_name

set_task_name('whu')
import os
from lib.utils.snake import snake_coco_utils, snake_config, visualize_utils
import cv2
import numpy as np
import math
from lib.utils import data_utils
import torch.utils.data as data
from pycocotools.coco import COCO
from lib.config import cfg


class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split):
        '''
        coco数据集解析
        :param ann_file: 注记json文件地址
        :param data_root: 图片文件夹目录
        :param split: 可取train(将进行缩放、等数据增强操作) mini(截取前500个数据）
        '''
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)  # 解析获得coco对象
        self.anns = sorted(self.coco.getImgIds())  # 获得images_id数组并排序
        self.anns = np.array([ann for ann in self.anns if len(self.coco.getAnnIds(imgIds=ann, iscrowd=0))])  # 去除没有注记的图片id
        self.anns = self.anns[:500] if split == 'mini' else self.anns  # split = mini时截取
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}

    def process_info(self, img_id):
        '''
        获取图片的信息
        :param img_id:
        :return: annotation数组，图片路径，图片id
        '''
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=0)  # 根据img_id获取此图片的所有annotation_id
        anno = self.coco.loadAnns(ann_ids)  # 根据id获取annotations数组
        path = os.path.join(self.data_root, self.coco.loadImgs(int(img_id))[0]['file_name'])
        return anno, path, img_id

    def custom_img_ids(self, img_ids):
        '''
        自定义选中的img_list
        :param img_ids:
        :return:
        '''
        self.anns = img_ids

    def read_original_data(self, anno, path):
        '''
        获取图片、多边形数组、类别数组
        :param anno:
        :param path:
        :return: 图片ndarray、多边形数组，各多边形对应的类别
        '''
        img = cv2.imread(path)  # 获取图片ndarray h,w,c
        # 获取轮廓多边形数组ndarray(N, 2) N个点，每个点有x，y坐标
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno]
        cls_ids = [self.json_category_id_to_contiguous_id[obj['category_id']] for obj in anno]
        return img, instance_polys, cls_ids

    def transform_original_data(self, instance_polys, flipped, width, trans_output, inp_out_hw):
        '''
        根据（图片增强的函数）变换多边形轮廓
        :param instance_polys:
        :param flipped:
        :param width:
        :param trans_output:
        :param inp_out_hw:
        :return: 变换后的多边形
        '''
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = [poly.reshape(-1, 2) for poly in instance]

            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = snake_coco_utils.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]  # 筛选端点数目大于4的多边形
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)  # 多边形的x坐标设置在多边形宽度以内
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)  # 多边形y坐标设置在多边形高度以内
            polys = snake_coco_utils.filter_tiny_polys(instance)  # 过滤面积小于5的多边形
            polys = snake_coco_utils.get_cw_polys(polys)  # 将多边形转为逆时针
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            polys = [poly for poly in polys if len(poly) >= 4]  # 去除端点数目小于4的多边形
            instance_polys_.append(polys)
        return instance_polys_

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = [snake_coco_utils.get_extreme_points(poly) for poly in instance]
            extreme_points.append(points)
        return extreme_points

    def prepare_detection(self, box, poly, ct_hm, cls_id, wh, reg, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        # box的中心点
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct_float = ct.copy()
        ct = np.round(ct).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])
        reg.append((ct_float - ct).tolist())

        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        return decode_box

    def prepare_init(self, box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, h, w):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        # x_min, y_min, x_max, y_max = box
        img_init_poly = snake_coco_utils.get_init(box)   # bbox中点连线组成菱形
        img_init_poly = snake_coco_utils.uniformsample(img_init_poly, snake_config.init_poly_num)  # 均匀重采样为40个点
        can_init_poly = snake_coco_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
        img_gt_poly = extreme_point
        can_gt_poly = snake_coco_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        i_it_4pys.append(img_init_poly)
        c_it_4pys.append(can_init_poly)
        i_gt_4pys.append(img_gt_poly)
        c_gt_4pys.append(can_gt_poly)

    def prepare_evolution(self, poly, extreme_point, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        octagon = snake_coco_utils.get_octagon(extreme_point)  # 极点生成八边形
        img_init_poly = snake_coco_utils.uniformsample(octagon, snake_config.poly_num)  # 八边形重采样为128个点
        can_init_poly = snake_coco_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)  # 平移

        '''下面的操作是将img_gt_poly的第一个点移动到与img_init_poly的第一个点距离最小的位置
        即将img_gt_poly 与 img_init_poly对齐，以减少计算量'''
        img_gt_poly = snake_coco_utils.uniformsample(poly, len(poly) * snake_config.gt_poly_num)
        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        can_gt_poly = snake_coco_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    def __getitem__(self, index):
        ann = self.anns[index]

        # coco注记对象，图片地址，image_id
        anno, path, img_id = self.process_info(ann)
        # 图片ndarray，轮廓多边形数组ndarray(N,2)，类别数组（一个多边形一个类别）
        img, instance_polys, cls_ids = self.read_original_data(anno, path)

        height, width = img.shape[0], img.shape[1]
        # 图片增强
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_coco_utils.augment(
                img, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                snake_config.mean, snake_config.std, instance_polys
            )
        # 按照图片增强的函数变换多边形
        instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)
        # 去除越界、编辑过小的多边形，转为逆时针
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)
        # 获取四个边界点，即deformation（菱形经snake到extreme）的目标值
        extreme_points = self.get_extreme_points(instance_polys)

        # detection
        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([cfg.heads.ct_hm, output_h, output_w], dtype=np.float32)
        wh = []
        reg = []
        ct_cls = []
        ct_ind = []

        # init
        i_it_4pys = []
        c_it_4pys = []
        i_gt_4pys = []
        c_gt_4pys = []

        # evolution
        i_it_pys = []
        c_it_pys = []
        i_gt_pys = []
        c_gt_pys = []

        gt_pys = []
        for i in range(len(anno)):
            cls_id = cls_ids[i]  # 类别数组
            instance_poly = instance_polys[i]  # 轮廓多边形数组
            instance_points = extreme_points[i]  # 边界点数组

            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                extreme_point = instance_points[j]

                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue

                # 遍历更新ct_hm 、 wh
                decode_box = self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, reg, ct_cls, ct_ind)
                # 遍历更新 init 的4个数组
                self.prepare_init(decode_box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
                # 遍历更新 evolution 的 4个数组
                self.prepare_evolution(poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)
                gt_pys.append(poly)

        ret = {'inp': inp}
        detection = {'ct_hm': ct_hm, 'wh': wh, 'reg': reg, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        # i_it_4py 40点，c_it_4py 40点，i_gt_4py 4点， c_gt_4py 4点
        init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
        # i_it_py 128点  c_it_py 128点， i_gt_py 128点（ground truth)，  c_gt_py 128点 （ground truth)
        evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
        ret.update(detection)
        ret.update(init)
        ret.update(evolution)
        # visualize_utils.visualize_snake_detection(orig_img, ret)
        # visualize_utils.visualize_snake_poly_init(orig_img, ret, 'i_it_4py')
        # visualize_utils.visualize_snake_evolution(orig_img, ret, 'i_gt_4py')
        # visualize_utils.visualize_snake_poly_init(orig_img, ret)
        # visualize_utils.visualize_snake_evolution(orig_img, ret)

        ct_num = len(ct_ind)  # contour的数目
        meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'ct_num': ct_num}
        if cfg.test_km:
            meta.update({'gt_py': gt_pys})
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.anns)


if __name__ == "__main__":
    dataset_path = "/home/ai-center/clover/project/building-extraction/snake-master/data/whumix-val/"
    ann_file = os.path.join(dataset_path, "val.json")
    data_root = os.path.join(dataset_path, "image")  # 图片地址
    dataset = Dataset(ann_file, data_root, "train")
    print(dataset)
    aa = dataset.__getitem__(0)

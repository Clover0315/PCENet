import numpy as np
from lib.config import cfg


mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.28863828, 0.27408164, 0.27809835],
               dtype=np.float32).reshape(1, 1, 3)
data_rng = np.random.RandomState(123)
eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                   dtype=np.float32)
eig_vec = np.array([
    [-0.58752847, -0.69563484, 0.41340352],
    [-0.5832747, 0.00994535, -0.81221408],
    [-0.56089297, 0.71832671, 0.41158938]
], dtype=np.float32)

down_ratio = 4
scale = np.array([800, 800])
input_w, input_h = (800, 800)
scale_range = np.arange(0.6, 1.4, 0.1)

voc_input_h, voc_input_w = (512, 512)
voc_scale_range = np.arange(0.6, 1.4, 0.1)

box_center = False
center_scope = False

init = 'quadrangle'
rect_poly_num = 40
init_poly_num = 40
ex_gt_num = cfg.ex_gt_num  # 极点数目
pan_center = cfg.pan_center  # 平移坐标到形状中心
poly_num = 128
gt_poly_num = poly_num
spline_num = 10

adj_num = 4  # snake的卷积核
state_dim = 128  # snake 原始128
fusion_state_dim = 256  # 原始256
state_dims = (state_dim * 8 + fusion_state_dim, 256, 64)  # snake
# 第一个是state_dim * (self.res_layer_num + 1) + fusion_state_dim

train_pred_box = False
box_iou = 0.7  # //0.7
confidence = 0.1
train_pred_box_only = True

train_pred_ex = False
train_nearest_gt = True

ct_score = cfg.ct_score

ro = 4

segm_or_bbox = 'segm'

ep_score = 0.6

dla_pretrained = True
dla_path = '/home/ai-center/clover/project/building-extraction/snake-master/data/model/backbone/dla34-ba72cf86.pth'
# dla_path = '/home/ai-center/clover/project/building-extraction/snake-master/data/model/backbone/dla34up-cityscapes-ed5cc4e8.pth'
# dla_path = '/home/ai-center/clover/project/building-extraction/snake-master/data/model/backbone/dla34+tricks-24a49e58.pth'
# dla_path = '/home/ai-center/clover/project/building-extraction/snake-master/data/model/backbone/dla102-d94d9790.pth'
# dla_path = '/home/ai-center/clover/project/building-extraction/snake-master/data/model/backbone/dla60-24839fc4.pth'



import os

from .yacs import CfgNode as CN
cfg = CN()

# model
cfg.model = 'whu'
cfg.model_dir = 'data/model/34/1'  # 模型存储文件夹

# network
cfg.network = 'ro_34'
cfg.dla_model_dir = 'data/model/dla/whu'  # 预训练模型dlaseg地址

# ct_hm + wh 解码得到中心点坐标，bbox，得分，类别
cfg.heads = CN({'ct_hm': 1, 'wh': 2})

# task
cfg.task = 'pcenet'
cfg.ex_gt_num = 4  # 极点数目
cfg.pan_center = False
cfg.freeze_dla = True  # 冻结dla参数
# ----------------------
# 顶点分类
cfg.test_km = False

# gpus
cfg.gpus = [0]

# if load the pretrained network
cfg.resume = True

cfg.segm_or_bbox = 'segm'
# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'WhuTrain'
cfg.train.epoch = 120
cfg.train.num_workers = 8

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 5e-4

cfg.train.warmup = False
cfg.train.scheduler = ''
cfg.train.milestones = [80, 120, 150, 170]
cfg.train.gamma = 0.5

cfg.train.batch_size = 32

# test
cfg.test = CN()
cfg.test.dataset = 'WhuVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1

# recorder
cfg.record_dir = 'data/record'

# result
cfg.result_dir = 'data/result'

# evaluation
cfg.skip_eval = False  # 不做cocoeval

cfg.save_ep = 10
cfg.eval_ep = 1

cfg.use_gt_det = True  # 使用ground truth 的detection

# -----------------------------------------------------------------------------
# snake
# -----------------------------------------------------------------------------
cfg.ct_score = 0.2  # 用于筛选detection 原值为0.05 buildmapper 是0.2
cfg.demo_path = '/home/ai-center/clover/project/building-extraction/snake-master/data/whumix-val/image/'

# assign the gpus
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

# assign the network head conv
cfg.head_conv = 64 if 'res' in cfg.network else 256

cfg.model_dir = os.path.join(cfg.model_dir, cfg.task, cfg.model)
cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.model)
cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.model)
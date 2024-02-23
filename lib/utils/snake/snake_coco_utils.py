from lib.utils.snake.snake_cityscapes_utils import *

input_scale = np.array([512, 512])


def augment(img, split, _data_rng, _eig_val, _eig_vec, mean, std, polys):
    '''
    数据增强
    :param img:  图片ndarray(使用cv imread()获取）
    :param split:
    :param _data_rng:
    :param _eig_val:
    :param _eig_vec:
    :param mean:
    :param std:
    :param polys:
    :return:
    '''
    # resize input
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  # 图片中心
    # 格式化图片大小为长宽相等
    scale = max(height, width)
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    # random crop and flip augmentation
    flipped = False  # 是否翻转
    if split == 'train':
        scale = scale * np.random.uniform(0.6, 1.4)  # 随机缩放大小（0.6~1.4倍）
        seed = np.random.randint(0, len(polys))  # 随机获取第seed个多边形：取0~len(polys）[多边形数目] 之间的一个随机整数
        index = np.random.randint(0, len(polys[seed][0]))  # 随机获取第seed个多边形的第index个顶点：取0~第seed个多边形的顶点数目 之间的一个整数
        x, y = polys[seed][0][index]  # 第seed个多边形第index个顶点的坐标
        center[0] = x
        border = scale[0] // 2 if scale[0] < width else width - scale[0] // 2
        center[0] = np.clip(center[0], a_min=border, a_max=width-border)
        center[1] = y
        border = scale[1] // 2 if scale[1] < height else height - scale[1] // 2
        center[1] = np.clip(center[1], a_min=border, a_max=height-border)

        # flip augmentation  随机翻转
        if np.random.random() < 0.5:
            flipped = True
            img = img[:, ::-1, :]
            center[0] = width - center[0] - 1

    input_w, input_h = input_scale
    if split != 'train':
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        x = 32
        input_w = (int(width / 1.) | (x - 1)) + 1
        input_h = (int(height / 1.) | (x - 1)) + 1
        scale = np.array([input_w, input_h])
        # input_w, input_h = (width + x - 1) // x * x, (height + x - 1) // x * x
        # input_w, input_h = int((width / 0.5 + x - 1) // x * x), int((height / 0.5 + x - 1) // x * x)
        # input_w, input_h = 512, 512
    # 获取仿射变换矩阵
    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
    # 图片变换（scale center ）
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    # color augmentation  颜色增强
    orig_img = inp.copy()
    inp = (inp.astype(np.float32) / 255.)
    if split == 'train':
        data_utils.color_aug(_data_rng, inp, _eig_val, _eig_vec)
        # data_utils.blur_aug(inp)

    # normalize the image  图片标准化
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)  # 改成 C H W用于后续conv

    output_h, output_w = input_h // snake_config.down_ratio, input_w // snake_config.down_ratio
    trans_output = data_utils.get_affine_transform(center, scale, 0, [output_w, output_h])
    inp_out_hw = (input_h, input_w, output_h, output_w)

    # orig_img：除颜色增强以外的增强图片， trans_input：有颜色增强的图片，
    # trans_input 放射变形矩阵[图片大小(512,512)]，trans_output 放射变形矩阵[图片大小(128,128)]
    # flipped 是否翻转，center 增强的中心 scale 缩放大小
    return orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw



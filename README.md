# PCENet
A progressive  method for extracting vector contour of buildings based on remote sensing images, aims to directly and accurately output the polygon contour of buildings through neural networks

### 一、安装环境：<a href="INSTALL.md">点此处</a>
### 获取训练集

### 修改配置
  - lib/config/config_whu.py ：whu-mix 数据集训练配置
  - lib/config/config_aicrowd.py:aicrowd 数据集训练配置
  - lib/config/config_ep_tp.py: 轮廓简化
### 训练模型：
```python
python main.py # 启动训练
```
### 测试结果：
```python
python test_imgs.py
```

# 效果图

# 技术支持
本文基础网络结构来自[deepsnake](https://github.com/zju3dv/snake)：

@inproceedings{peng2020deep,
  title={Deep Snake for Real-Time Instance Segmentation},
  author={Peng, Sida and Jiang, Wen and Pi, Huaijin and Li, Xiuli and Bao, Hujun and Zhou, Xiaowei},
  booktitle={CVPR},
  year={2020}
}
### Set up the python environment

```
# 使用conda创建python环境
conda create -n pcenet python=3.8
conda activate pcenet
```
多驱动配置可参考[使用conda安装的cudatoolkit安装NVIDIA](https://blog.csdn.net/j___t/article/details/103882584?spm=1001.2014.3001.5506)，在conda环境中安装nvidia驱动
```
# 安装pytorch1.8.1和对应的cuda环境
pip install Cython==0.28.2
pip install -r requirements.txt

# install apex
cd
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 39e153a3159724432257a8fc118807b359f4d1c8
python setup.py
```

### Compile cuda extensions under `lib/csrc`
可参考[DCNv2在pytorch1.8.1的安装](https://blog.csdn.net/super_lxc/article/details/132895812)
```
ROOT=/path/to/snake
cd $ROOT/lib/csrc
cd dcn_v2
python setup.py build_ext --inplace
cd ../extreme_utils
python setup.py build_ext --inplace
cd ../roi_align_layer
python setup.py build_ext --inplace
```

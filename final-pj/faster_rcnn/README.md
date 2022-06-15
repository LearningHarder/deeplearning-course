# Faster R-CNN

该项目主要是来自开源仓库 https://github.com/WZMIAOMIAO/deep-learning-for-image-processing.git 中pytorch_object_detection 中的faster rcnn,使用的backbone为resnet50


## 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.7.1(注意：必须是1.6.0或以上，因为使用官方提供的混合精度训练1.6.0后才支持)
* pycocotools(Linux:`pip install pycocotools`; Windows:`pip install pycocotools-windows`(不需要额外安装vs))
* Ubuntu或Centos(不建议Windows)
* 最好使用GPU训练
* 详细环境配置见`requirements.txt`

## 文件结构：
```
  ├── backbone: 特征提取网络，可以根据自己的要求选择
  ├── network_files: Faster R-CNN网络（包括Fast R-CNN以及RPN等模块）
  ├── train_utils: 训练验证相关模块（包括cocotools）
  ├── my_dataset.py: 自定义dataset用于读取VOC数据集
  ├── train_mobilenet.py: 以MobileNetV2做为backbone进行训练
  ├── train_resnet50_fpn.py: 以resnet50+FPN做为backbone进行训练
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── train_multi_GPU_random.py: 随机初始化参数
  ├── train_multi_GPU_imagenet.py: 使用imageNet初始化backbone网络参数
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
  └── pascal_voc_classes.json: pascal_voc标签文件
```

## 预训练权重下载地址（下载后放入backbone文件夹中）：
* ResNet50 backbone:  https://download.pytorch.org/models/resnet50-0676ba61.pth
* 注意，下载的预训练权重记得要重命名，比如在 train_multi_GPU_imagenet.py中读取的是`resnet50.pth`文件，
 
 
 
## 数据集，本例程使用的是PASCAL VOC2007数据集
* Pascal VOC2007 train/val数据集下载
```
!wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
```


## 训练方法(基于colab)
* 克隆仓库
```
!git clone https://github.com/LearningHarder/deeplearning-course.git
%cd deeplearning-course/mid-pj/faster_rcnn
```
* 确保提前准备好数据集
```
!wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
!tar -xf VOCtrainval_06-Nov-2007.tar
```
* 下载好对应预训练模型权重
```
!wget -c "https://download.pytorch.org/models/resnet50-0676ba61.pth" -O ./backbone/resnet50.pth
```
* 随机初始化训练
```
python -m torch.distributed.launch --nproc_per_node=2  --use_env train_multi_GPU_random.py --epochs 70 --lr 0.02 -b 6
```
* imageNet初始化
```
python -m torch.distributed.launch --nproc_per_node=1  --use_env train_multi_GPU_imagenet.py --epochs 40 --lr 0.01 -b 6
```
* coco+mask_rcnn初始化


* `CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py`

## 注意事项


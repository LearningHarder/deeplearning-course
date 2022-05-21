# deeplearning-course

## Table of Contents

- [AlexNet](#AlexNet)
- [Faster R-CNN](#Faster R-CNN)
- [YOLO V3](#YOLO V3)
- [template](#template)


## AlexNet
任务一使用AlexNet的模型，测试Alexnet模型对CIFAR-100数据集的分类效果，在此基础上与使用cutout, mixup, cutmix这几种数据增强方法下的数据集训练效果进行了对比。
使用说明如下：
**1.git clone cifar100-aug/cifar100-baseline**
**2.训练模型 python main.py **
注明：模型的训练过程中的相关参数会记录在相对路径的log文件夹中，可使用tensorboard可视化，图片的可视化文件见数据探索与可视化notebook文件。

## Faster R-CNN

## YOLO V3
本项目中YOLO V3模型的训练基于[官方开源项目](https://github.com/ultralytics/yolov3) 。使用步骤如下。

**1.git clone 官方项目**

```
git clone https://github.com/ultralytics/yolov3
cd yolov3
pip3 install -r requirements.txt
```
**2.修改部分文件**
将val.py，data/voc.yaml，和untils/loggers/__init__.py更新为此仓库的对应文件，将此仓库的plots.py文件拷贝到对应位置
**3.训练模型**
```
python train.py --data data/voc.yaml --epochs 80 --batch-size 16 --img 416
```

**4.绘制曲线**
```
python plots.py
```
**5.检测VOC数据集以外的图片**
data/images路径下保存了待检测的图片
```
python detect.py --weight runs/train/exp/weights/best.pt --save-txt
```

## template

This module depends upon a knowledge of [Markdown]().

```
```

### Any optional sections




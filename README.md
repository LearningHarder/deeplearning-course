# deeplearning-course

## Table of Contents

- [AlexNet](#AlexNet)
- [Faster R-CNN](#Faster R-CNN)
- [YOLO V3](#YOLO V3)
- [template](#template)


## AlexNet

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




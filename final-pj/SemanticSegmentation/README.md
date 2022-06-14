# Semantic Segmentation on AD video
我们使用开源的github项目 https://github.com/open-mmlab/mmsegmentation.git ，在colab环境进行实验
使用的模型为DeepLabV3+ ，载入在cityscapes数据集上训练所得的权重，测试的视频为b站up主**柠檬气泡水Nicole**2022年5月30日上传的约2分钟的行车视频素材https://www.bilibili.com/video/BV1v94y1S7so?spm_id_from=333.337.search-card.all.click
```
!wget https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_cityscapes/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth
```
取其中一帧，语义分割效果如下：

![image](https://github.com/LearningHarder/deeplearning-course/blob/master/final-pj/SemanticSegmentation/testimg.jpg)

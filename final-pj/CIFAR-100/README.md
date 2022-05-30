# Vision-Transformer On CIFAR-100
In this part, we implement a model based on the structure of Vision Transformer, and compare the accuracy with Alexnet with similar parameters size. And we try several data augmentation method to compare with the baseline of both Convolutionary Neural Networks model and Transformer model.

## Quick Usage with Google Colab
#### Vision Transformer
1) Git Clone the Repository and go to the corresponding file folder.
```!git clone https://github.com/LearningHarder/deeplearning-course.git```
```!cd final-pj/ViT```
2) Install all the libraries required.
```pip install -r ViT/requirements``` 
3) Train the ViT from scratch.
```python3 train.py```
4) Give Predictions if need. 
 ```python3 predict.py``` .
5) Plot Accuracy and Loss through epoches. 
 ```!``` .
Notes: It is possible that colab can't install albumentation in the requirements.txt. If so, please delete albumentation in the requirement.txt and use pip to install directly.
 ```!pip install albumentations ```

 
## Results 

The ViT took about __ minutes to train on CIFAR-100 dataset. Without any prior pre-training we were able to achieve about 74% accuracy on validation dataset.

<!-- <p align="center">
  <img src="https://github.com/ShivamRajSharma/Vision-Transformer/blob/master/acc_plot.png" height="300"/>
</p> -->

## Model Comparison
<pre>
Alexnet v.s. ViT
1) Numbers of Parameters:  : Training the whole network from scratch.
1) Training Stratergy      : Training the whole network from scratch.
2) Optimizer               : Adam optimizer was used with weight decay.
3) Learning Rate Scheduler : Linear decay with warmup.
4) Regularization          : Dropout, HorizontalFlip, RandomBrightness, RandomContrast, RGBShift, GaussNoise
5) Loss                    : Categorical Cross-Entropy Loss.
6) Performance Metric      : Accuracy.
7) Performance             : 74% Accuracy
7) Epochs Trained          : 80
8) Training Time           : 12 minutes
</pre>


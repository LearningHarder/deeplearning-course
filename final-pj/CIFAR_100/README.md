# Vision-Transformer On CIFAR-100
In this part, we implement a model based on the structure of Vision Transformer, and compare the accuracy with Alexnet with similar parameters size. And we try several data augmentation method to compare with the baseline of both Convolutionary Neural Networks model and Transformer model.

## Quick Usage with Google Colab
#### Vision Transformer & Alexnet
1) Git Clone the Repository and go to the corresponding file folder.
```!git clone https://github.com/LearningHarder/deeplearning-course.git```
2) Install all the libraries required.
```!pip install -r requirements.txt``` 
3) Train the ViT from scratch.
```python train.py```
4) Plot Accuracy and Loss through epoches with TensorBoard
 ```%load_ext tensorboard```
```%tensorboard --logdir log``` .

 
## Results 

The ViT took about __ minutes to train on CIFAR-100 dataset. Without any prior pre-training we were able to achieve about 74% accuracy on validation dataset.


## Model  Info 
<pre>
Alexnet v.s. ViT
1) Numbers of Parameters:  : 2.88M 
1) Training Stratergy      : Training the whole network from scratch.
2) Optimizer               : Adam optimizer with weight decay.
3) Learning Rate Scheduler : Linear decay with warmup.
4) Regularization          : Dropout, HorizontalFlip, RandomBrightness, RandomContrast, RGBShift, GaussNoise
5) Loss                    : Categorical Cross-Entropy Loss.
6) Performance Metric      : Accuracy.
7) Performance             : % Accuracy v.s.
7) Epochs Trained          : 80
8) Training Time           : 12 minutes
</pre>


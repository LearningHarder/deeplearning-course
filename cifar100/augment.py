import torch
import numpy as np
import copy

class Cutout(object):
    """随机使用0填充若干个色块.
    Args:
        n_holes (int):填充的块数.
        length (int): 每一块所占的pixel数
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length
        

    def __call__(self, train_batch):
        """对一个batch进行cutout的操作
        Args:
            img_batch List(tuple(Tensor)): 由若干个Tensor image(C, H, W)和对应的label(int)构成的一个batch.
        Returns:
            img_batch_cutout List(tuple(Tensor): 由若干个Tensor image(C, H, W)
            labels List(tuple(Tensor): 对应的label
        """

        def cutout_one(img):
            """对一张照片进行cutout
            Args:
                img (Tensor): 一张图片(C, H, W).
            Returns:
                Tensor: 遮盖后的图片数据
            """
            h = img.size(1)
            w = img.size(2)
            mask = np.ones((h, w), np.float32)
            for n in range(self.n_holes):
                y = np.random.randint(h)  # 返回随机数/数组(整数)
                x = np.random.randint(w)
                y1 = np.clip(y - self.length // 2, 0, h) #截取函数
                y2 = np.clip(y + self.length // 2, 0, h) #⽤于截取数组中⼩于或者⼤于某值的部分，
                x1 = np.clip(x - self.length // 2, 0, w) #并使得被截取的部分等于固定的值
                x2 = np.clip(x + self.length // 2, 0, w)
                mask[y1: y2, x1: x2] = 0.
            mask = torch.from_numpy(mask)   #数组转换成张量，且⼆者共享内存，对张量进⾏修改⽐如重新赋值，那么原始数组也会相应发⽣改变
            mask = mask.expand_as(img)  #把⼀个tensor变成和函数括号内⼀样形状的tensor
            img = img * mask
            return img

        img_batch_cutout = []
        labels = []
        for each in train_batch:
            img = cutout_one(each[0])
            label = np.eye(100)[each[1]]
            img_batch_cutout.append(img)
            labels.append(torch.from_numpy(label))
        return torch.stack(img_batch_cutout,0),torch.stack(labels,0)


class Mixup(object):
    """对一个batch进行shuffle后，对每张图片都更新lam
    Args:
        alpha (float): 生成beta分布的参数alpha，设取值为0.5
    """
    def __init__(self, alpha):
        self.alpha = alpha
        
    def __call__(self, train_batch):
        """
        Args:
            img_batch List(tuple(Tensor,int)): 由若干个Tensor image(C, H, W)和对应的label(int)构成的一个batch.
        Returns:
            img_batch_mixup List(tuple(Tensor,np.array) 由若干个Tensor image(C, H, W)
            labels List(tuple(Tensor): 对应的label
        """
        alpha = self.alpha
        rand_index = np.random.permutation(len(train_batch))
        img_batch_mixup = []
        labels = []
        for i in range(len(train_batch)):
            lam = np.random.beta(alpha,alpha)
            mix_img = train_batch[i][0] * lam + train_batch[rand_index[i]][0] * (1-lam)
            mix_label = lam * np.eye(100)[train_batch[i][1]] + (1- lam) * lam * np.eye(100)[train_batch[rand_index[i]][1]]            
            img_batch_mixup.append(mix_img)
            print(type(mix_img,mix_label))
            labels.append(torch.from_numpy(mix_label))
        return torch.stack(img_batch_mixup,0),torch.stack(labels,0)



class CutMix(object):
    """对一个batch进行shuffle后，一一对应后，将同剪裁并将对应的图的位置的像素数据放进该图中
    Args:
        alpha (float): 生成beta分布的参数alpha，设取值为0.5
    """
    def __init__(self, alpha):
        self.alpha = alpha
        
    def __call__(self, train_batch):
        """
        Args:
            img_batch List(tuple(Tensor,int)): 由若干个Tensor image(C, H, W)和对应的label(int)构成的一个batch.
        Returns:
            img_batch_cutmix List(Tensor), int: 由若干个Tensor image(C, H, W)
            labels List(Tensor): 对应的label
        """

        def rand_patch(size, lam):
            """
            Args:
                size Torch.Size: 是Tensor image(C, H, W)的size值
                lam int :取alpha=0.5后用beta分布生成的lambda值
            Returns:
                px1,py1,px2,py2 int: 剪切的4个点的位置
            """
            H = size[1]
            W = size[2]
            
            cut_rat = np.sqrt(1. - lam)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)

            cx = np.random.randint(W)
            cy = np.random.randint(H)

            px1 = np.clip(cx - cut_w // 2, 0, W)
            py1 = np.clip(cy - cut_h // 2, 0, H)
            px2 = np.clip(cx + cut_w // 2, 0, W)
            py2 = np.clip(cy + cut_h // 2, 0, H)

            return px1, py1, px2, py2

        alpha = self.alpha
        rand_index = np.random.permutation(len(train_batch))
        image_batch_edit = copy.deepcopy(train_batch)
        img_batch_cutmix = []
        labels = []

        for i in range(len(train_batch)):
            lam = np.random.beta(alpha, alpha)
            px1, py1, px2, py2 = rand_patch(train_batch[0][0].shape, lam)
            image_batch_edit[i][:,px1:px2, py1:py2] = train_batch[rand_index[i]][:,px1:px2, py1:py2]
            lam_pic = 1 - ((px2 - px1) * (py2 - py1) / (image_batch_edit[i].shape[1] * image_batch_edit[i].shape[2]))
            mix_label = lam_pic * lam * np.eye(100)[train_batch[i][1]] + (1- lam_pic) * lam * np.eye(100)[train_batch[rand_index[i]][1]]
            img_batch_cutmix.append(image_batch_edit[i])
            labels.append(torch.from_numpy(mix_label))
        return torch.stack(img_batch_cutmix,0),torch.stack(labels,0)
import torch
import numpy as np
import torchvision.datasets as datasets
from skimage.color import rgb2lab, rgb2gray


def count_params(model):
    '''
    returns the number of trainable parameters in a model
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GrayscaleImageFolder(datasets.ImageFolder):
    '''
    Custom dataloader for converting images to grayscale before loading.
    '''

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img_orig = self.transform(img)
            img_orig = np.asarray(img_orig)
            img_lab = rgb2lab(img_orig)
            img_lab = (img_lab+128)/255
            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
            img_orig = rgb2gray(img_orig)
            img_orig = torch.from_numpy(img_orig).unsqueeze(0).float()

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_orig, img_ab, target

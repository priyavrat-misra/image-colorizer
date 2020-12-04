import torch
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, rgb2gray, lab2rgb


def count_params(model):
    '''
    returns the number of trainable parameters in some model
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GrayscaleImageFolder(datasets.ImageFolder):
    '''
    Custom dataloader for various operations on images before loading them.
    '''

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img_orig = self.transform(img)  # apply transforms
            img_orig = np.asarray(img_orig)  # convert to numpy array
            img_lab = rgb2lab(img_orig)  # convert RGB image to LAB
            img_ab = img_lab[:, :, 1:3]  # separate AB channels from LAB
            img_ab = (img_ab + 128) / 255  # normalize the pixel values
            # transpose image from HxWxC to CxHxW and turn it into a tensor
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
            img_orig = rgb2gray(img_orig)  # convert RGB to grayscale
            # add a channel axis to grascale image and turn it into a tensor
            img_orig = torch.from_numpy(img_orig).unsqueeze(0).float()

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_orig, img_ab, target


def load_gray(path, max_size=360, shape=None):
    '''
    load an image as grayscale, change the shape as per input,
    perform transformations and convert it to model compatable shape.
    '''
    img_gray = Image.open(path).convert('L')

    if max(img_gray.size) > max_size:
        size = max_size
    else:
        size = max(img_gray.size)

    if shape is not None:
        size = shape

    img_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    img_gray = img_transform(img_gray).unsqueeze(0)
    return img_gray


def to_rgb(img_l, img_ab):
    '''
    concatinates Lightness (grayscale) and AB channels,
    and converts the resulting LAB image to RGB
    '''
    if img_l.shape == img_ab.shape:
        img_lab = torch.cat((img_l, img_ab), 1).numpy().squeeze()
    else:
        img_lab = torch.cat(
            (img_l, img_ab[:, :, :img_l.size(2), :img_l.size(3)]),
            dim=1
        ).numpy().squeeze()

    img_lab = img_lab.transpose(1, 2, 0)  # transpose image to HxWxC
    img_lab[:, :, 0] = img_lab[:, :, 0] * 100  # range pixel values from 0-100
    img_lab[:, :, 1:] = img_lab[:, :, 1:] * 255 - 128  # un-normalize
    img_rgb = lab2rgb(img_lab.astype(np.float64))  # convert LAB image to RGB

    return img_rgb

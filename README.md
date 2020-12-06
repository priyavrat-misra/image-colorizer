# Image Colorization (WIP)

> ![cover](https://github.com/priyavrat-misra/image-colorization/blob/master/images/colorized/bnw_col.png?raw=true)
> _<sup>Figure. Grayscale image (left); Colorized image (right)</sup>_
## Overview
> This project is a Deep Convolutional Neural Network approach to solve the task of image colorization.
> The goal is to produce a colored image given a grayscale image. <br>
> At it's heart, it uses Convolutional Auto-Encoders to solve this task.
> First few layers of [ResNet-18](https://arxiv.org/abs/1512.03385) model are used as the Encoder,
> and the Decoder consists of a series of Deconvolution layers (i.e., upsample layers followed by convolutions) and residual connections.<br>
> The model is trained on a subset of [MIT Places365](http://places2.csail.mit.edu/index.html) dataset, consisting of `41000` images of landscapes and scenes.


## README.md
- [x] Overview
- [ ] Approach
- [ ] Steps
- [ ] Results
- [ ] Dependencies
- [ ] Setup
- [ ] Usage

## Todo
- [x] define & train a model architecture
- [x] add argparse support
- [x] define a more residual architecture
- [x] use pretrained resnet-18 params for the layers used in the encoder & train the model
- [x] check how the colorization effect varies with image resolution
- [x] separate the model from the checkpoint file to a different file
- [ ] deploy with flask
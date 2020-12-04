# Image Colorization (WIP)

## Overview
> This project is a Deep Convolutional Neural Network approach to solve the task of image colorization.
> The goal is to produce a colored image given a grayscale image. <br>
> At it's heart, it uses Convolutional Auto-Encoders to solve this task.
> First few layers of [ResNet-18](https://arxiv.org/abs/1512.03385) model are used as the Encoder,
> and the Decoder consists of a series of convolution layers, residual connections and nearest-neighbor upsample operations. <br>
> The model is trained on a subset of [MIT Places365](http://places2.csail.mit.edu/index.html) dataset, consisting of `41000` RGB images of landscapes and scenes.

## Results
> ![cover-1](https://github.com/priyavrat-misra/image-colorization/blob/master/images/outputs/bnw_col_3.png?raw=true)
> ![cover-2](https://github.com/priyavrat-misra/image-colorization/blob/master/images/outputs/bnw_col_2.png?raw=true)
> ![cover-3](https://github.com/priyavrat-misra/image-colorization/blob/master/images/outputs/bnw_col_1.png?raw=true)
> _<sup>Grayscale images (left); Colorized images (right)</sup>_

## README.md
- [x] Overview
- [ ] Approach
- [ ] Steps
- [x] Results
- [ ] Dependencies
- [ ] Setup
- [ ] Usage

## Todo
- [x] define and train a model architecture
- [x] add argparse support
- [ ] define and train with a more residual architecture
- [ ] deploy with flask
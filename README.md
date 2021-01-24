# Image Colorization

![cover](https://github.com/priyavrat-misra/image-colorization/blob/master/images/colorized/bnw_col.png?raw=true "an eye-candy")

## Contents
- [Overview](#overview)
- [Approach](#approach)
- [Steps](#steps)
- [Results](#results)
- [TL;DR](#tldr)
- [Setup](#setup)
- [Usage](#usage)
- [Todo](#todo)

<br>

## Overview
> This project is a Deep Convolutional Neural Network approach to solve the task of image colorization.
> The goal is to produce a colored image given a grayscale image.<br>
> At it's heart, it uses Convolutional Auto-Encoders to solve this task.
> First few layers of [ResNet-18](https://arxiv.org/abs/1512.03385) model are used as the Encoder,
> and the Decoder consists of a series of Deconvolution layers (i.e., upsample layers followed by convolutions) and residual connections.<br>
> The model is trained on a subset of [MIT Places365](http://places2.csail.mit.edu/index.html) dataset, consisting of `41000` images of landscapes and scenes.

## Approach
> The images in the dataset are in RGB Colorspace.
> Before loading the images, the images are converted to [LAB colorspace](https://en.wikipedia.org/wiki/CIELAB_color_space).
> This colorspace contains exactly the same information as RGB.<br>
> It has 3 channels, `Lightness, A and B`.
> The lightness channel can be used as the grayscale equivalent of a colored image,
> the rest 2 channels (A and B) contain the color information.<br>
>
> In a nutshell, the training process follows these steps:
>> 1. The lightness channel is separated from the other 2 channels and used as the model's input.
>> 2. The model predicts the A and B channels (or 'AB' for short).
>> 3. The loss is calculated by comparing the predicted AB and the corresponding original AB of the input image.
>
> More about the training process can be found [here](https://github.com/priyavrat-misra/image-colorization/blob/master/train.ipynb "train.ipynb").

## Steps
> 1. [Defining a model architecture:](https://github.com/priyavrat-misra/image-colorization/blob/master/network.py "network.py")
>    - The model follows an Auto-Encoder kind of architecture i.e., it has an `encoder` and a `decoder` part.
>    - The encoder is used to _extract features_ of an image whereas,
>    - the decoder is used to upsample the features. In other words, it increases the _spacial resolution_.
>    - In here, the layers of the encoder are taken from ResNet-18 model, and the first conv layer is modified to take a single channel as input (i.e., grayscale or lightness) rather than 3 channels.
>    - The decoder uses nearest neighbor upsampling (for increasing the spacial resolution),
>     followed by convolutional layers (for dealing with the depth).
>    - A more detailed visualization of the model architecture can be seen [here](https://github.com/priyavrat-misra/image-colorization/blob/master/images/architecture.png?raw=true 'after all "A picture is worth a thousand words" :)').
> 2. [Defining a custom dataloader:](https://github.com/priyavrat-misra/image-colorization/blob/master/utils.py "utils.GrayscaleImageFolder")
>    - when loading the images, it converts them to LAB, and returns L and AB separately.
>    - it does few data processing tasks as well like applying tranforms and normalization.
> 3. [Training the model:](https://github.com/priyavrat-misra/image-colorization/blob/master/train.ipynb "train.ipynb")
>    - The model is trained for 64 epochs with [Adam Optimization](https://arxiv.org/abs/1412.6980).
>    - For calculating the loss between the predicted AB and the original AB, Mean Squared Error is used.
> 4. [Inference:](https://github.com/priyavrat-misra/image-colorization/blob/master/inference.ipynb "inference.ipynb")
>    - Inference is done with unseen images and the results look promising, or should I say "natural"? :)

## Results
> ![results](https://github.com/priyavrat-misra/image-colorization/blob/master/images/results.png?raw=true)
> _<sup>More colorized examples can be found in [here](https://github.com/priyavrat-misra/image-colorization/blob/master/images/colorized/).<sup>_

## TL;DR
> Given an image, the model can colorize it.

## Setup
- Clone and change directory:
```bash
git clone "https://github.com/priyavrat-misra/image-colorization.git"
cd image-colorization/
```
- Dependencies:
```bash
pip install -r requirements.txt
```

## Usage
```bash
python colorize.py --img-path <path/to/image.jpg> --out-path <path/to/output.jpg> --res 360
# or the short-way:
python colorize.py -i <path/to/image.jpg> -o <path/to/output.jpg> -r 360
```

_Note:_
> - As the model is trained with 224x224 images, it gives best results when `--res` is set to lower resolutions (<=480) and okay-ish when set around ~720.
> - Setting `--res` higher than that of input image won't increase the output's quality.

<br>

## Todo
- [x] define & train a model architecture
- [x] add argparse support
- [x] define a more residual architecture
- [x] use pretrained resnet-18 params for the layers used in the encoder & train the model
- [x] check how the colorization effect varies with image resolution
- [x] separate the model from the checkpoint file to a different file
- [x] complete README.md
- [ ] deploy with flask
- [ ] _after that, host it maybe?_

<br>

For any queries, feel free to reach me out on [LinkedIn](https://linkedin.com/in/priyavrat-misra/).
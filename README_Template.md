# COVIDNet
>An ensembled deep neural network model for predicting COVID-19 with chest x-rays.

[![NPM Version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Downloads Stats][npm-downloads]][npm-url]

| Network Model | Original Paper | 
|     :---:     |     :---:      |
| DenseNet-121     | [Densely Connected Convolutional Networks](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)|
| EfficientNet-B2| [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)|
| Inception-V3| [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf)|
| Inception-ResNet-V2| [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf)|
| ResNet-50   | [Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) |
| Xception| [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf)|

![](header.png)

## Environment

### Dependencies
- tensorflow==
- keras==
- 

## Table of Contents

- [Train COVIDNet from Scratch](#Train-COVIDNet-from-Scratch)
  * [Preprocessing](#Preprocessing)
    + [Download Unet Weights](#Download-Unet-Weights)
    + [Crop images](#Crop-images)
  * [Pretrain with NIH dataset](#Pretrain-with-NIH-dataset)
  * [Find best hyper parameters](#Find-best-hyper-parameters)
  * [Train model with best parameters](#Train-model-with-best-parameters)
  * [Ensemble models](#Ensemble-models)
- [Test CovidNet on your own dataset](#How-to-Test-COVIDNet-on-Your-Own-Dataset)
  * [Download trained weights](#Download-the-well-trained-weights)
- [Grad-CAM Visualization](#Grad-CAM-Visualization)
- [Citation](#citation)

## Train COVIDNet from Scratch


### Preprocessing 

Prepare cropped images.

#### Download Unet Weights
We used Unet to segment the input image. The link to download the weights is: [trained_model.hdf5](https://github.com/imlab-uiip/lung-segmentation-2d/blob/master/trained_model.hdf5)

#### Crop images
```sh
python preprocess.py -f [IMAGE FOLDER PATH] -U [trained_model.hdf5 PATH] -o [IMAGE OUTPUT PATH]
```
Put images in one folder.

```sh
python preprocess.py -h
```
to get more details.


### Pretrain with NIH dataset
```sh
pretrain.py [-h] [-m MODEL_NAME] [-s IMG_SIZE] [-p NIH_PATH]
```

```sh
usage: pretrain.py [-h] -m model_name -s img_size -p NIH_path

Pretrain a model on NIH dataset.

optional arguments:
  -h, --help            show this help message and exit
  -m model_name, -model model_name
                        the name of the model to be trained with NIH dataset.
                        Choose from ResNet-50, Xception, DenseNet-121,
                        Inception-V3,Inception-ResNet-V2, EfficientNet-B2.
  -s img_size, --size img_size
                        the size of NIH images
  -p NIH_path, --path NIH_path
                        the path that contains NIH dataset and NIH csv file or
                        the path in which a new directory for NIH dataset will
                        be created.
```

### Find best hyper parameters
```sh
tuner.py [-h] [-m MODEL_NAME] [-s IMG_SIZE] [-p DATA_PATH]
```

```sh
usage: tuner.py [-h] -m model_name --size img_size --path DATA_path

Use keras tuner to find best hyper parameter.

optional arguments:
  -h, --help            show this help message and exit
  -m model_name, --model model_name
                        the name of the model to be trained. Choose from
                        ResNet-50, Xception, DenseNet-121,
                        Inception-V3,Inception-ResNet-V2, EfficientNet-B2
  --size img_size, -s img_size
                        the size of dataset images
  --path DATA_path, -p DATA_path
                        the path that contains the dataset.
```


### Train model with best parameters

### Ensemble models

## How to Test COVIDNet on Your Own Dataset

### Download the well-trained weights
[Google drive link to trained weights](https://drive.google.com/drive/folders/1_FRViB9xnX1-8582WGfXquOLn2YuiR3k?usp=sharing)

## Grad-CAM Visualization 

## Release History

* 0.2.1
    * CHANGE: Update docs (module code remains unchanged)
* 0.2.0
    * CHANGE: Remove `setDefaultXYZ()`
    * ADD: Add `init()`
* 0.1.1
    * FIX: Crash when calling `baz()` (Thanks @GenerousContributorName!)
* 0.1.0
    * The first proper release
    * CHANGE: Rename `foo()` to `bar()`
* 0.0.1
    * Work in progress

## Meta

Your Name – [@YourTwitter](https://twitter.com/dbader_org) – YourEmail@example.com

Distributed under the XYZ license. See ``LICENSE`` for more information.

[https://github.com/yourname/github-link](https://github.com/dbader/)

## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki

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

## How to Train COVIDNet from Scratch


### 1. Preprocessing (prepare the cropped image)

#### Download Unet Weights
We used Unet to segment the input image. The link to download the weights is: [trained_model.hdf5](https://github.com/imlab-uiip/lung-segmentation-2d/blob/master/trained_model.hdf5)

#### Crop the Image Folder
```sh
python preprocess.py -f [IMAGE FOLDER PATH] -U [trained_model.hdf5 PATH] -o [IMAGE OUTPUT PATH]
```
Put images in one folder.

```sh
python preprocess.py -h
```
to get more details.


### 2. Pretrain a model with NIH dataset
```sh
python pretrain.py --model model_name NIH_path --create
```
### 3. Train and test each individual model

### 4. Ensemble the model

## How to Test COVIDNet on Your Own Dataset

### Download the well-trained weights
Google drive link: 

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

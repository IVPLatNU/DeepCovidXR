# DeepCOVID-XR
>An ensemble convolutional neural network (CNN) model for detecting frontal chext x-rays (CXRs) suspicious for COVID-19.

While findings on chest imaging are not sensitive nor specific enough to replace diagnostic testing for COVID-19, artificially intelligent (AI) systems for automated analysis of CXRs have a potential role in triage and infection control within a hospital setting. Much of the current evidence regarding AI platforms for analysis of CXRs is limited by very small datasets and/or publicly available data of questionable quality. DeepCOVID-XR is a weighted ensemble of six popular CNN architectures - DenseNet-121, EfficientNet-B2, Inception-V3, Inception-ResNet-V2, ResNet-50 and Xception - trained and tested on a large clinical dataset from a major US healthcare system, the largest clinical dataset of CXRs from the COVID-19 era to date. 

## Dataset
DeepCOVID-XR is pretrained on 112,120 images from the NIH CXR-14 dataset (NIH dataset is publicly available and can be downloaded by running pretrian.py. The dataset contains  frontal CXR images that are labeled with 14 separate disease classifications. The algorithm was then fine tuned on over 14,000 clinical images (>4,000 COVID-19 positive) from the COVID-19 era and tested on a hold out dataset of over 2,000 images (>1,000 COVID-19 positive) from a hold-out institution that the model was not exposed to during training. 

## Performance 
Deep-COVIDXR correctly classified images as COVID-19 positive or COVID-19 negative using RT-PCR for the SARS-COV2 virus as the reference standard with an accuracy of 85% and an area under the ROC curve of 0.899, outperforming 3 individual experienced thoracic radiologists and with performance similar to that of the consensus of all 3 radiologists.  

## Model
![ensembled model](/img/model.jpg)
| Network Model | Original Paper | 
|     :---:     |     :---:      |
| DenseNet-121     | [Densely Connected Convolutional Networks](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)|
| EfficientNet-B2| [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)|
| Inception-V3| [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf)|
| Inception-ResNet-V2| [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf)|
| ResNet-50   | [Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) |
| Xception| [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf)|
| U-Net | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)|

![](header.png)

## Environment

### Dependencies
- pandas==0.25.0
- nibabel==3.1.0
- efficientnet==1.1.0
- scikit_image==0.15.0
- tqdm==4.46.0
- keras_vis==0.5.0 [(installed directly from github link)](https://github.com/raghakot/keras-vis)
- keras_tuner==1.0.1
- opencv_python==4.2.0.34
- matplotlib==3.2.1
- numpy==1.17.0
- Keras==2.3.1
- tf_nightly_gpu_2.0_preview==2.0.0.dev20190814
- deepstack==0.0.9
- Pillow==7.2.0
- scikit_learn==0.23.2
- skimage==0.0
- tensorflow==2.3.0
- vis==0.0.5

To set up the environment and install all the packages, run
```sh
$pip install -r requirements.txt
```

## Table of Contents

- [Train DeepCOVID-XR from Scratch](#Train-DeepCOVID-XR-from-Scratch)
  * [Preprocessing](#Preprocessing)
    + [Download Unet Weights](#Download-Unet-Weights)
    + [Crop images](#Crop-images)
    + [Resize images](#Resize-images)
  * [Pretrain with NIH dataset](#Pretrain-with-NIH-dataset)
  * [Find best hyper parameters](#Find-best-hyper-parameters)
  * [Train model with best parameters](#Train-model-with-best-parameters)
  * [Ensemble models](#Ensemble-models)
- [Test DeepCOVID-XR on individual image](#Test-DeepCOVID-XR-on-individual-image)
  * [Download trained weights](#Download-the-well-trained-weights)
- [Grad-CAM Visualization](#Grad-CAM-Visualization)
- [Citation](#citation)

## Train DeepCOVID-XR from Scratch

To train DeepCOVID-XR, the dataset should be structured as below:

```sh
├───224
│   ├───crop
│   │   ├───Test
│   │   │   ├───Negative
│   │   │   └───Positive
│   │   ├───Train
│   │   │   ├───Negative
│   │   │   └───Positive
│   │   └───Validation
│   │       ├───Negative
│   │       └───Positive
│   └───uncrop
│       ├───Test
│       │   ├───Negative
│       │   └───Positive
│       ├───Train
│       │   ├───Negative
│       │   └───Positive
│       └───Validation
│           ├───Negative
│           └───Positive
└───331
    ├───crop
    │   ├───Test
    │   │   ├───Negative
    │   │   └───Positive
    │   ├───Train
    │   │   ├───Negative
    │   │   └───Positive
    │   └───Validation
    │       ├───Negative
    │       └───Positive
    └───uncrop
        ├───Test
        │   ├───Negative
        │   └───Positive
        ├───Train
        │   ├───Negative
        │   └───Positive
        └───Validation
            ├───Negative
            └───Positive
```

The cropped version of dataset can be obtained with preprocessing.

### Preprocessing 

Prepare cropped and resized images.

#### Download Unet Weights
We used Unet to segment the input image. The link to download the weights is: [trained_model.hdf5](https://github.com/imlab-uiip/lung-segmentation-2d/blob/master/trained_model.hdf5)

#### Crop images
```sh
python Crop_img.py -f [IMAGE FOLDER PATH] -U [trained_model.hdf5 PATH] -o [IMAGE OUTPUT PATH]
```
Please put images in one folder. Use '-h' to get more details.

```sh
python Crop_img.py -h
```

#### Resize images
```sh
python Resize_img.py -i [IMAGE INPUT PATH] -o [IMAGE OUTPUT PATH] -s [RESIZE SHAPE (331 or 244)]
```
After preprocessing step a and b, you would get 224_crop/224_uncrop/331_crop/331_uncrop dataset respectively.

### Pretrain with NIH dataset

A base model without dropout layer can be trained with NIH dataset. The resulting weight will be saved and used for further training.
If the NIH dataset already exists, you can provide a path to the dataset for training. If NIH does not exist, provide a path you would like to download the dataset in.
You also need to provide the name of the model to be trained and the size of the image you would like to train with. 

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

Keras tuner will be used to find best values for learning rate, momentum and dropout rate for dropout layers. 
All layers except for the global average pooling layer be frozen at first. And then the entire model will be unfrozon and used to find best hyper parameters.
Note: nih_path is the directory that contains the pretrained nih weight file.

```sh
tuner.py [-h] [-m MODEL_NAME] [-s IMG_SIZE] [-p DATA_PATH] [-n nih_path]
```

```sh
usage: tuner.py [-h] -m model_name --size img_size --path DATA_path --nihpath
                NIH_weight_path

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
  --nihpath NIH_weight_path, -n NIH_weight_path
                        the path to pretrained NIH weight file.
```


### Train model with best parameters

Each model can be trained on a custom dataset. All layers except for the global average pooling layer be frozen at first.
And then the entire model will be unfrozen and trained. The resulting weight will be further used for ensembling.  
Note: nih_path is the directory that contains the pretrained nih weight file.

```sh
train.py [-h] [-m MODEL_NAME] [-s IMG_SIZE] [-p DATA_PATH] [-n nih_path]
```

```sh
usage: train.py [-h] -m model_name --size img_size --path DATA_path --nihpath
                NIH_weight_path

Train a model on a given dataset.

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
  --nihpath NIH_weight_path, -n NIH_weight_path
                        the path to pretrained nih weight.
```

### Ensemble models

```sh
ensemble.py [-h] [-w WEIGHTS_PATH] [-d DATA_PATH]
```

```sh
usage: ensemble.py [-h] --weight weight_path --data CROPPED_DATA_path

Ensemble trained models to generate confusion matrices.

optional arguments:
  -h, --help            show this help message and exit
  --weight weight_path, -w weight_path
                        the path that contains trained weights.
  --data CROPPED_DATA_path, -d CROPPED_DATA_path
                        the path that contains the entire dataset.
```

## Test DeepCOVID-XR on individual image

Provided a single image, a prediction can be made toward the possibility of covid positive using the ensembled model. 

```sh
test.py [-h] [-w WEIGHTS_PATH] [-i IMAGE_PATH]
```

```sh
usage: test.py [-h] --weight weight_path --image IMAGE_path

For each input image, generates COVID possibilities for 224x224 and 331x331
versions.

optional arguments:
  -h, --help            show this help message and exit
  --weight weight_path, -w weight_path
                        the path that contains trained weights.
  --image IMAGE_path, -i IMAGE_path
                        the path to the image.
```

### Download the well-trained weights
[Google drive link to trained weights](https://drive.google.com/drive/folders/1_FRViB9xnX1-8582WGfXquOLn2YuiR3k?usp=sharing)

## Grad-CAM Visualization 

Grad-CAM heat maps can be generated for an individual image of a folder contains several images. 
An example is provided as below

```sh
test.py [-h] [-w WEIGHTS_PATH]
```

```sh
usage: visSample.py [-h] --weight weight_path

For each input image, generates and saves a grad-CAM image.

optional arguments:
  -h, --help            show this help message and exit
  --weight weight_path, -w weight_path
                        the path that contains trained weights.
```

![grad-CAM](/img/covid_positive.png)


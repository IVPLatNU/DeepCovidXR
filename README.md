# DeepCOVID-XR
>An ensemble convolutional neural network (CNN) model for detecting frontal chest x-rays (CXRs) suspicious for COVID-19.

While findings on chest imaging are not sensitive nor specific enough to replace diagnostic testing for COVID-19, artificially intelligent (AI) systems for automated analysis of CXRs have a potential role in triage and infection control within a hospital setting. Much of the current evidence regarding AI platforms for analysis of CXRs is limited by very small datasets and/or publicly available data of questionable quality. 

DeepCOVID-XR is a weighted ensemble of six popular CNN architectures - DenseNet-121, EfficientNet-B2, Inception-V3, Inception-ResNet-V2, ResNet-50 and Xception - trained and tested on a large clinical dataset from a major US healthcare system, to our knowledge the largest clinical dataset of CXRs from the COVID-19 era used to train a published AI system to date. 

For those looking for the pre-trained model weights only, they can be downloaded here:
 [Google drive link to trained weights](https://drive.google.com/drive/folders/1_FRViB9xnX1-8582WGfXquOLn2YuiR3k?usp=sharing)
 
 **!! Note: This platform is not FDA approved for clinical use and is only intended to be used for research purposes. See our license for more details.**

## Model
![ensembled model](/img/Model.png)
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

Note the trained weights of each of the CNN members of the weighted ensemble are available [here](https://drive.google.com/drive/folders/1_FRViB9xnX1-8582WGfXquOLn2YuiR3k?usp=sharing). The trained weights for averaging predictions of each of the models for ensembling purposes are available [here](/ensemble_weights.pickle). The instructions below walk through the entire process of training a model from scratch and also provide code for using our already trained weights for analyzing external datasets and/or individual images. 

## Dataset
DeepCOVID-XR was first pretrained on 112,120 images from the NIH CXR-14 dataset. The NIH dataset is publicly available and can be downloaded [here](https://nihcc.app.box.com/v/ChestXray-NIHCC). The dataset contains  frontal CXR images that are labeled with 14 separate disease classifications. 

The algorithm was then fine tuned on over 14,000 clinical images (>4,000 COVID-19 positive) from the COVID-19 era and tested on a hold out dataset of over 2,000 images (>1,000 COVID-19 positive) from a hold-out institution that the model was not exposed to during training. 

## Table of Contents

- [Train DeepCOVID-XR from Scratch](#Train-DeepCOVID-XR-from-Scratch)
  * [Environment](#Environment)
  * [Preprocessing](#Preprocessing)
    + [Download Unet Weights](#Download-Unet-Weights)
    + [Crop images](#Crop-images)
    + [Resize images](#Resize-images)
  * [Pretrain with NIH dataset](#Pretrain-with-NIH-dataset)
  * [Find best hyper parameters](#Find-best-hyper-parameters)
  * [Train model with best parameters](#Train-model-with-best-parameters)
  * [Ensemble models](#Ensemble-models)
  * [Evaluate ensemble model](#Evaluate-ensemble-model)
- [DeepCOVID-XR Prediction On New Data](#DeepCOVID-XR-Prediction-On-New-Data)
  * [Trained model weights](#Trained-model-weights)
  * [Ensemble weights](#Ensemble-weights)
- [Grad-CAM Visualization](#Grad-CAM-Visualization)

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

The cropped and resized versions of dataset can be obtained with preprocessing. 
If there is significant class imbalance in your dataset, you should consider oversampling the minority class to improve training. 
Further details on oversampling can be found [here](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#oversample_the_minority_class).

### Environment
This package was created within an environment specified by building a docker container on a Centos7 server with 5 NVIDIA TitanV GPUs. 

The OS/drivers/software used to create this package were as follows:

Ubuntu 18.04.2 LTS

NVIDIA driver version: 410.93

CUDA version: 10.0.130

CUDNN version: 7.6.2

Python version == 3.6.8

We recommend using python version 3.6 or 3.7, as we have not tested this application on other versions. Additionally, the dependencies required to run this library are listed below:

#### Dependencies
- pandas==0.25.0
- nibabel==3.1.0
- efficientnet==1.1.0 https://github.com/qubvel/efficientnet
- scikit_image==0.15.0
- tqdm==4.46.0
- keras_vis==0.5.0 [(installed directly from github link)](https://github.com/raghakot/keras-vis)
- keras_tuner==1.0.1 https://keras-team.github.io/keras-tuner/
- opencv_python==4.2.0.34 
- matplotlib==3.2.1
- numpy==1.17.0
- Keras==2.3.1
- tensorflow-gpu==2.0.0 
- deepstack==0.0.9 https://github.com/jcborges/DeepStack
- Pillow==7.2.0
- scikit_learn==0.23.2
- skimage==0.0
- tensorflow==2.0.0 
- vis==0.0.5

To set up the environment:

Option1:

Pull a docker image from DockerHub that contains all of the necessary dependencies. deepcovidxr:large is our production image and includes packages that are not necessary for this project. Warning: you must have 17.3GB free disk space to pull this docker image. 

```sh
docker pull rwehbe/deepcovidxr:large
```

You can also pull a smaller version of a docker image that includes only dependencies necessary for this project. Note: this has not been extensively tested as has the larger image. Warning: you must have at least 5.48GB free disk space to pull this docker image.

```sh
docker pull rwehbe/deepcovidxr:small
```

[Link to DockerHub Repository](https://hub.docker.com/repository/docker/rwehbe/deepcovidxr/general)

Option2:

To install all the packages in a conda environment or virtualenv, run

```sh
$pip install -r requirements.txt
```

Note: This has not been as extensively tested as using the Docker images above. You may run into issues depending on your hardware/OS specifics. 

### Preprocessing 
Note: All input data should be in 8-bit png or jpeg format. DICOM/Nifti files should be converted to 8-bit png or jpeg files via appropriate preprocessing and windowing based on metadata.

The data is first cropped in order to produce a version of the image that focuses on the lung fields. A square cropping region is used in order to retain important extraparenchymal anatomy (pleural spaces) and retrocardiac anatomy (left lower pulmonary lobe). Both cropped and uncropped images serve as inputs into the ensemble algorithm. Each image (cropped and uncropped) is then downsampled using Lanczos resmapling to two different sizes, 224X224 pixels and 331X331 pixels, for a total of 4 images as input into each of the CNN members of the weighted ensemble.

Prepare cropped and resized images.

#### Download Unet Weights
We used a Unet to segment the input image, then crop a square region surrounding the lung fields. The UNet model used in preprocessing is based on that developed here: [https://github.com/imlab-uiip/lung-segmentation-2d/]. The link to download the weights is: [trained_model.hdf5](https://github.com/imlab-uiip/lung-segmentation-2d/blob/master/trained_model.hdf5). These are also provided in our Github repo.

#### Crop images
```sh
python crop_img.py -f [IMAGE FOLDER PATH] -U [trained_model.hdf5 PATH] -o [IMAGE OUTPUT PATH]
```
Please put images in one folder. Use '-h' to get more details.

```sh
python crop_img.py -h
```

#### Resize images
```sh
python resize_img.py -i [IMAGE INPUT PATH] -o [IMAGE OUTPUT PATH] -s [RESIZE SHAPE (331 or 244)]
```
After cropping and resizing images, the resulting datasets will be 224_crop/224_uncrop/331_crop/331_uncrop, respectively.

### Pretrain with NIH dataset
A base model without dropout layer can be trained with NIH dataset. The resulting weights will be saved and used for further training.
If the NIH dataset already exists, you can provide a path to the dataset for training. If NIH does not exist, provide a path you would like to download the dataset to.

You also need to provide the name of the model to be trained and the size of the images you would like to train with. 

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
Keras tuner will be used to find best values for learning rate, momentum and dropout rate for a single dropout layer before the output layer. 
All layers except for the output layer will be frozen initially, then the entire model will be unfrozen and used to find best hyper parameters.
Note: nih_path is the directory that contains the pretrained nih weight file. This module outputs weights of highest performing hypermodels, as well as a pickled dictionary containing values of the best performing hyperparameters.

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
Each model can be trained on a custom dataset. All layers except for the final output layer will be frozen at first.
And then the entire model will be unfrozen and trained. Note: nih_path is the directory that contains the pretrained nih weight file. Hyperparameters is the path to the pickled dictionary of the best performing hyperparameter values for the model being trained.

```sh
train.py [-h] [-m MODEL_NAME] [-s IMG_SIZE] [-p DATA_PATH] [-w weight_path] [-o output_path] [-h hyperparameters_path]
```

```sh
usage: train.py [-h] -m model_name --size img_size --path DATA_path --output
                prediction_output_path --weight_path weight_path
                [--hyperparameters Hyperparameters]

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
  --output prediction_output_path, -o prediction_output_path
                        the directory to output training curves and saved
                        weights
  --weight_path weight_path, -w weight_path
                        the path to pretrained weights, either NIH if training
                        from scratch or corresponding model weights from our
                        pretrained weights if fine-tuning DeepCOVID-XR.
  --hyperparameters Hyperparameters, -hy Hyperparameters
                        the path to pickled hyperparameters dictionary; will
                        use default parameters if not provided.

```


### Ensemble models
This module computes the weights that will be multiplied by individual model predictions for a weighted average ensemble prediction using a Bayesian model combination approach. 
You can provide custom trained weights for each member of the model ensemble to calculate the optimal weights for each member prediction. 

Note: If using your own weights, folder provided should contain weight files for each of the 24 trained trained ensemble models in the format '{Model}_{img_size}\_up\_{crop_stat}.h5', where model is one of:
(ResNet50, Xception, Inception, DenseNet, InceptionResNet, EfficientNet), img_size is one of (224, 331), and crop_stat is one of: (crop, uncrop).

```sh
ensemble.py [-h] [-w WEIGHTS_PATH] [-d DATA_PATH] [-o OUTPUT_PATH]
```

```sh
usage: ensemble.py [-h] --weight weight_path --data DATA_path
                   [--output prediction_output_path]

Ensemble trained models to generate confusion matrices.

optional arguments:
  -h, --help            show this help message and exit
  --weight weight_path, -w weight_path
                        the path that contains trained weights.
  --data DATA_path, -d DATA_path
                        the path that contains the entire dataset.
  --output prediction_output_path, -o prediction_output_path
                        the directory to output a csv file of predictions and
                        pickled list of ensemble weights; if not provided will
                        save to current working directory
```


### Evaluate ensemble model
Produces a csv file of predictions vs. actual labels for a test dataset organized in a subdirectory tree according to the schema provided above.
Also generates a confusion matrix and ROC curve.

Note: If using your own weights, folder provided should contain weight files for each of the 24 trained trained ensemble models in the format '{Model}_{img_size}\_up\_{crop_stat}.h5', where model is one of:
(ResNet50, Xception, Inception, DenseNet, InceptionResNet, EfficientNet), img_size is one of (224, 331), and crop_stat is one of: (crop, uncrop).

```sh
evaluate.py [-h] [-w WEIGHTS_PATH] [-i IMAGE_PATH]
```

```sh
usage: evaluate.py [-h] --weight weight_path --image IMAGE_path -o
                   prediction_output_path
                   [--ensemble_weight ensemble_weight_path] [--tta]

For each input image, generates predictions of COVID-19 status.

optional arguments:
  -h, --help            show this help message and exit
  --weight weight_path, -w weight_path
                        the path that contains trained weights.
  --image IMAGE_path, -i IMAGE_path
                        the path to the image directory with subdirectory tree
                        as indicated in the README file
  -o prediction_output_path, --output prediction_output_path
                        the directory to save results, including a csv of
                        predictions, confusion matrix, and ROC curve
  --ensemble_weight ensemble_weight_path, -e ensemble_weight_path
                        the path to the ensemble weights as a pickled list, if
                        not supplied uses our pretrained weights by default
  --tta, -t             switch to turn on test-time augmentation, warning:
                        this takes significantly longer as each model
                        prediction is run 10 times
```


## DeepCOVID-XR Prediction On New Data
Produces predictions of COVID-19 positivity or negativity on a folder of images or a single image. Note this module performs all preprocessing steps including cropping and resizing - therefore original png images can be provided for analysis. 
After analysis is complete, the user will be prompted as to whether further predictions are desired. 

Note: If using your own weights, folder provided should contain weight files for each of the 24 trained trained ensemble models in the format '{Model}_{img_size}\_up\_{crop_stat}.h5', where model is one of:
(ResNet50, Xception, Inception, DenseNet, InceptionResNet, EfficientNet), img_size is one of (224, 331), and crop_stat is one of: (crop, uncrop).

Note: GPU use is recommended. Loading model weights takes some time, but once loaded predictions on a single image take a matter of seconds on a single NVIDIA Titan V GPU. 

```sh
predict.py [-h] -w weight_path -i IMAGE_path
                  [-e ensemble_weight_path] [-t tta]
                  [-o prediction_output_path]
```

```sh
usage: predict.py [-h] --weight weight_path --image IMAGE_path
                  [--ensemble_weight ensemble_weight_path] [--tta]
                  [--output prediction_output_path]

For each input image, generates predictions of COVID-19 status.

optional arguments:
  -h, --help            show this help message and exit
  --weight weight_path, -w weight_path
                        the path that contains trained weights.
  --image IMAGE_path, -i IMAGE_path
                        the path to the image/folder of images.
  --ensemble_weight ensemble_weight_path, -e ensemble_weight_path
                        the path to the ensemble weights as a pickled list, if
                        not supplied uses our pretrained weights
  --tta, -t         switch to turn on test-time augmentation, warning:
                        this takes significantly longer as each model
```

### Trained model weights
Our trained model weights are provided so that DeepCOVIDXR can be be tested and/or fine tuned on external datasets.
[Google drive link to trained weights](https://drive.google.com/drive/folders/1_FRViB9xnX1-8582WGfXquOLn2YuiR3k?usp=sharing)

### Ensemble weights
Our ensemble weights for computing a weighted average of individual model predictions are provided as a pickled list [here](/ensemble_weights.pickle).
This list is ordered as follows:

`[ dense_224_uncrop,
                  dense_224_crop,
                  dense_331_uncrop,
                  dense_331_crop,
                  res_224_uncrop,
                  res_224_crop,
                  res_331_uncrop,
                  res_331_crop,
                  inception_224_uncrop,
                  inception_224_crop,
                  inception_331_uncrop,
                  inception_331_crop,
                  inceptionresnet_224_uncrop,
                  inceptionresnet_224_crop,
                  inceptionresnet_331_uncrop,
                  inceptionresnet_331_crop,
                  xception_224_uncrop,
                  xception_224_crop,
                  xception_331_uncrop,
                  xception_331_crop,
                  efficient_224_uncrop,
                  efficient_224_crop,
                  efficient_331_uncrop,
                  efficient_331_crop] `

## Grad-CAM Visualization 
Note: GPU use is recommended. Loading model weights takes some time, but once loaded generating Grad-CAM heatmaps takes a matter of seconds per image on a GPU.
Grad-CAM heat maps can be generated for an individual image of a folder contains several images. 
An example is provided as below

```sh
visSample.py [-h] [-w WEIGHTS_PATH] [-p IMAGE_PATH]
```

```sh
usage: visSample.py [-h] --weight weight_path --path path to image/folder of images

For each input image, generates and saves a grad-CAM image.

optional arguments:
  -h, --help            show this help message and exit
  --weight weight_path, -w weight_path
                        the path that contains trained weights.
  --path image_path, -p image_path
                        the path to the image or folder of images being analyzed.
```

![grad-CAM](/img/covid_positive.png)


# Train an individual model
import argparse
import os
from utils import imgUtils, trainFeatures
from covid_models import DenseNet, ResNet, XceptionNet, EfficientNet, InceptionNet, InceptionResNet 

# Implement command line argument
parser = argparse.ArgumentParser(description='Use keras tuner to find best hyper parameter.')

parser.add_argument('-m', '--model', dest='model_name', metavar = 'model_name', 
                    choices = ['ResNet-50', 'Xception', 'DenseNet-121', 'Inception-V3', 
                               'Inception-ResNet-V2', 'EfficientNet-B2'],
                    type = str, nargs = 1, required = True, 
                    help = 'the name of the model to be trained.\n Choose from ResNet-50, Xception, DenseNet-121, Inception-V3,' 
                            'Inception-ResNet-V2, EfficientNet-B2')

parser.add_argument('--size', '-s', dest='img_size', metavar = 'img_size',
                    type = int, nargs = 1, required = True, 
                    help = 'the size of dataset images')

parser.add_argument('--path', '-p', dest='path', metavar='DATA_path', type=str, nargs=1,
                    required = True, help='the path that contains the dataset.')
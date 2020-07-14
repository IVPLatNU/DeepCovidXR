# Pretrain a given model on NIH dataset
import argparse

from utils import imgUtils, trainFeatures
from covid_models import hyperModel, DenseNet, ResNet, XceptionNet, EfficientNet, InceptionNet, InceptionResNet 

# Implement command line argument
parser = argparse.ArgumentParser(description='Find best hyper parameters for a model')
parser.add_argument('-m', '--model', dest = 'model_name')

parser.add_argument('model', metavar = 'model_name', 
                    choices = ['ResNet-50', 'Xception', 'DenseNet-121', 'Inception-V3', 
                               'Inception-ResNet-V2', 'EfficientNet-B2'],
                    type = str, nargs = 1,
                    help = 'the name of the model to be trained.\n Choose from ResNet-50, Xception, DenseNet-121, Inception-V3,' 
                            'Inception-ResNet-V2, EfficientNet-B2')

parser.add_argument('path', metavar='DATA_path', type=str, nargs=1,
                    help='the path that contains the dataset.')

args = parser.parse_args()
data_path = args.path[0]
model_name = args.model_name

img_size = 331
batch_size = 16
rotation_range = 20
height_shift = 0.05
width_shift = 0.05

img_proc = imgUtils(img_size)
train_idg, val_idg = img_proc.dataGen(rotation_range, height_shift, width_shift)

# Train a given model on NIH dataset
lr = 0.001
momentum = 0.9
nestrov = True

patience_rlr = 2
patience_es = 10
factor = 0.1
min_delta = 0.001
monitor = 'val_auc'

if model_name == 'ResNet-50':
    resnet = ResNet('imagenet')
    model = resnet.buildBaseModel(img_size)
    resnet.compileModel(model, lr, momentum, nestrov)
    
elif model_name == 'Xception':
    xception = XceptionNet('imagenet')
    model = xception.buildBaseModel(img_size)
    xception.compileModel(model, lr, momentum, nestrov)
elif model_name == 'DenseNet-121':
    dense = DenseNet('imagenet')
    model = dense.buildBaseModel(img_size)
    dense.compileModel(model, lr, momentum, nestrov)
    
elif model_name == 'Inception-V3':
    inception = InceptionNet('imagenet')
    model = inception.buildBaseModel(img_size)
    inception.compileModel(model, lr, momentum, nestrov)
    
elif model_name == 'Inception-ResNet-V2':
    inceptionres = InceptionResNet('imagenet')
    model = inceptionres.buildBaseModel(img_size)
    inceptionres.compileModel(model, lr, momentum, nestrov)
    
elif model_name == 'EfficientNet-B2':
    efficient = EfficientNet('imagenet')
    model = efficient.buildBaseModel(img_size)
    efficient.compileModel(model, lr, momentum, nestrov)

features = trainFeatures()
rlp = features.setRLP(monitor, factor, patience_rlr)
es = features.setES(monitor, patience_es, min_delta)
cp = features.setCP(monitor, model_save_path)

epochs = 50
features.generator(model, batch_size, train_generator, val_generator, epochs, cp, rlp, es)

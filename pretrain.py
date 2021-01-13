# Pretrain a given model on NIH dataset
import argparse
import os
from utils import nihUtils, imgUtils, trainFeatures

""" NIH pretrain

This script will perform pretraining on one of the six models on the NIH dataset. 
The dataset will be downloaded and unzipped if it does not exist in the input path. 

The provided parameters are the model to be pretrained, the size of NIH images 
and the path to the NIH dataset. A .csv file named "NIH_Data_Entry.csv" and a directory
that contains the NIH images are required to start the pretrain. 

The pretrain hyperparameters can be found in the main function, which includes: 
    batch size (int), image augmentation parameters, learning rate (float), momentum (float),
    Nesterov momentum (boolean), early stopping parameters, 
The model will be trained with 50 epochs.
"""
def get_args():
    """
    This function retrieves user input command line arguments. 
    """
    parser = argparse.ArgumentParser(description='Pretrain a model on NIH dataset.')
    
    parser.add_argument('-m', '-model', dest='model_name', metavar = 'model_name', 
                        choices = ['ResNet-50', 'Xception', 'DenseNet-121', 'Inception-V3', 
                                   'Inception-ResNet-V2', 'EfficientNet-B2'],
                        type = str, nargs = 1, required = True, 
                        help = 'the name of the model to be trained with NIH dataset.\n Choose from ResNet-50, Xception, DenseNet-121, Inception-V3,' 
                                'Inception-ResNet-V2, EfficientNet-B2.')
    
    parser.add_argument('-s', '--size', dest='img_size', metavar = 'img_size',
                        type = int, nargs = 1, required = True,
                        help = 'the size of NIH images')
    
    parser.add_argument('-p', '--path', dest='path', metavar='NIH_path', type=str, nargs=1, default = '',
                        required = True, 
                        help='the path that contains NIH dataset and NIH csv file or the path in which a new '
                             'directory for NIH dataset will be created.')

    return parser.parse_args()

if __name__=='__main__':

    """
    This  is the main function for NIH dataset processing and pretraining. 
    """
    args = get_args()
    nih_path = os.path.normpath(args.path[0])
    img_size = args.img_size[0]
    model_name = args.model_name[0]

    batch_size = 16
    rotation_range = 15
    height_shift = 0.05
    width_shift = 0.05
    out_num = 16

    nih = nihUtils()
    nih_path, create_dir = nih.createDir(nih_path)
    model_save_path = 'nih_weight_{name}.h5'.format(name = model_name)
    
    if create_dir:
        nih.nihDownload(nih_path)
        
    nih_img_path = nih_path
    nih_csv_path = nih_path
    csv_name = os.path.join(nih_csv_path, 'NIH_Data_Entry.csv')

    train_df, val_df, labels = nih.nihSplit(csv_name, nih_img_path)
    
    label_len = len(labels)
    
    img_proc = imgUtils(img_size)
    train_idg, val_idg = img_proc.dataGen(rotation_range, height_shift, width_shift)
    
    train_generator, val_generator = nih.nihGenerator(img_size, 
                                                      batch_size, train_idg, val_idg, 
                                                      train_df, val_df, labels)

    
    # Pretrain a given model on NIH dataset
    lr = 0.001
    momentum = 0.9
    nestrov = True
    
    patience_rlr = 2
    patience_es = 10
    factor = 0.1
    min_delta = 0.001
    monitor = 'val_auc'
    
    features = trainFeatures()
    rlp = features.setRLP(monitor, factor, patience_rlr)
    es = features.setES(monitor, patience_es, min_delta)
    cp = features.setCP(monitor, model_save_path)
           
    _, model = features.getNihModel(model_name, img_size, 'imagenet', label_len)
    features.compileModel(model, lr, momentum, nestrov)
    print('Model compiled!')
    epochs = 50
    features.NIHgenerator(model, batch_size, train_generator, val_generator, epochs, cp, rlp, es)
    print('Done')

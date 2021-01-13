# Train an individual model
import argparse
import os
from utils import imgUtils, trainFeatures
import pickle

"""Train model

This script will train one of the six models used in ensemble with a given dataset.
The fully connected layers will be freezed at first. Then all layers will be unfreezed.
Hyperparameters can be changed in the main functions.

"""

def get_args():
    """
    This function gets various user input form command line. The user input variables
    include the name of the model to be trained, the size of the input images, 
    the path to the iamge dataset, the output path where the results and trained
    weights will be saved, the path to the weight from pretraining no NIH dataset
    and the path to pickled hyper parameters.
    
    Returns:
        parser.parse_args() (list): a list of user input values.
    """
    
    parser = argparse.ArgumentParser(description='Train a model on a given dataset.')
    
    parser.add_argument('-m', '--model', dest='model_name', metavar = 'model_name', 
                        choices = ['ResNet-50', 'Xception', 'DenseNet-121', 'Inception-V3', 
                                   'Inception-ResNet-V2', 'EfficientNet-B2'],
                        type = str, required = True,
                        help = 'the name of the model to be trained.\n Choose from ResNet-50, Xception, DenseNet-121, Inception-V3,' 
                                'Inception-ResNet-V2, EfficientNet-B2')
    
    parser.add_argument('--size', '-s', dest='img_size', metavar = 'img_size',
                        type = int, required = True,
                        help = 'the size of dataset images')
    
    parser.add_argument('--path', '-p', dest='path', metavar='DATA_path', type=str,
                        required = True, help='the path that contains the dataset.')

    parser.add_argument('--output', '-o', dest='output', metavar='prediction_output_path', type=str,
                        default=None, required=True, help='the directory to output training curves and saved weights')
    
    parser.add_argument('--weight_path', '-w', dest='weight_path', metavar='weight_path', type=str,
                        required = True, help='the path to pretrained weights, either NIH if training from scratch or '
                                              'corresponding model weights from our pretrained weights if fine-tuning'
                                              ' DeepCOVID-XR.')

    parser.add_argument('--hyperparameters', '-hy', dest='hyperparameters', metavar='Hyperparameters', type=str,
                        required=False, default=None, help='the path to pickled hyperparameters dictionary; will use '
                                                           'default parameters if not provided.')
    
    return parser.parse_args()

def make_path(data_dir, base, exp_name):
    
    """
    This function creates path to save the training results and weights.
    
    Parameters:
        data_dir (string): the path to the parent directory of training and 
        validation datasets.
        
        base (string): the path to the parent directory of saved weights.
        
        exp_name (string): a unique name for different experiment runs.
        
    Returns:
        train_path (string): the path to the training dataset.
        valid_path (string): the path to the validation dataset.
        freeze_weight_save_path (string): the path to the trained weight with layers 
        freezed.
        unfreeze_weight_save_path (string): the path to the trained weight with all
        layers unfreezed.
        freeze_img_save_path (string): the path to the result images with layers
        freezed. 
        unfreeze_img_save_path (string): the path to the result images with all
        layers unfreezed.
    """

    train_path = os.path.join(data_dir, 'Train')
    valid_path = os.path.join(data_dir, 'Validation')

    if (not os.path.isdir(train_path)) or (not os.path.isdir(valid_path)):
        print('Please split images into train directory and validation directory.')
        exit()

    freeze_weight_save_path = os.path.join(base, 'save_weights_initial', exp_name + '.h5')
    unfreeze_weight_save_path = os.path.join(base, 'save_weights_final', exp_name + '.h5')
    freeze_img_save_path = os.path.join(base, 'save_plots_initial', exp_name)
    unfreeze_img_save_path = os.path.join(base, 'save_plots_final', exp_name)

    if not os.path.exists(os.path.join(base, 'save_weights_initial')):
        os.makedirs(os.path.join(base, 'save_weights_initial'))
    
    if not os.path.exists(os.path.join(base, 'save_weights_final')):
        os.makedirs(os.path.join(base, 'save_weights_final'))

    if not os.path.exists(os.path.join(base, 'save_plots_initial')):
        os.makedirs(os.path.join(base, 'save_plots_initial'))

    if not os.path.exists(os.path.join(base, 'save_plots_final')):
        os.makedirs(os.path.join(base, 'save_plots_final'))
        
    if not os.path.exists(freeze_img_save_path):
        os.makedirs(freeze_img_save_path)
        
    if not os.path.exists(unfreeze_img_save_path):
        os.makedirs(unfreeze_img_save_path)
    
    return train_path, valid_path, freeze_weight_save_path, unfreeze_weight_save_path, freeze_img_save_path, unfreeze_img_save_path

if __name__=='__main__':
    
    """
    The main function sets vairous training parameters such as batch size
    and image augmentation parameters. The fully connected layers are first trained
    for 50 epochs and then the entire network is trained for another 50 epochs.
    
    The hyper parameters are set in this main function and can be changed below.
    The result images will be saved.
    """
    
    batch_size = 16
    rotation_range = 15
    height_shift = 0.05
    width_shift = 0.05
    
    args = get_args()

    data_path = os.path.normpath(args.path)
    model_name = args.model_name
    img_size = args.img_size
    weights = os.path.normpath(args.weight_path)
    hyperparameters = args.hyperparameters
    exp_name = model_name + '_' + str(img_size)
    output_path = args.output

    if output_path is not None:
        output_path = os.path.normpath(output_path)
    else:
        output_path = os.getcwd()

    train_dir, valid_dir, weight_dir1, weight_dir2, img_dir1, img_dir2 = make_path(data_path, output_path, exp_name)

    img_proc = imgUtils(img_size)
    train_idg, val_idg = img_proc.dataGen(rotation_range, height_shift, width_shift)
    train_gen, val_gen = img_proc.generator(batch_size, train_idg, val_idg, train_dir, valid_dir)

    if hyperparameters is not None:
        hyperparameters = pickle.load(open(hyperparameters, "rb"))
        lr = hyperparameters['learning_rate']
        momentum = hyperparameters['momentum']
        dropout_rate = hyperparameters['dropout_rate']
    else:
        lr = 0.001
        momentum = 0.9
        dropout_rate = 0.3

    nesterov = True
    patience_rlr = 3
    patience_es = 5
    factor = 0.1
    min_delta = 0.001
    monitor = 'val_auc'
    pre_epoch = 50
    epoch = 50

    features = trainFeatures()
    rlr = features.setRLP(monitor, factor, patience_rlr)
    es = features.setES(monitor, patience_es, min_delta)
    cp = features.setCP(monitor, weight_dir1)
    
    dropout_model = features.getDropoutModel(model_name, img_size, weights, dropout_rate)
    features.compileModel(dropout_model, lr, momentum, nesterov)
    model_history = features.generator(dropout_model, train_gen, val_gen, pre_epoch, cp, rlr, es)
    img_proc.plot_save(model_history, img_dir1)

    # Unfreeze and train the entire model
    model = features.load(dropout_model, weight_dir1)
    model = features.unfreeze(model)

    patience_es = 10
    
    es = features.setES(monitor, patience_es, min_delta)
    cp = features.setCP(monitor, weight_dir2)
    
    features.compileModel(model, lr, momentum, nesterov)
    model_history = features.generator(dropout_model, train_gen, val_gen, epoch, cp, rlr, es)
    img_proc.plot_save(model_history, img_dir2)
    print('Done')

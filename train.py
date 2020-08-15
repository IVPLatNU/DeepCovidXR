# Train an individual model
import argparse
import os
from utils import imgUtils, trainFeatures

def get_args():
    # Implement command line argument
    parser = argparse.ArgumentParser(description='Train a model on a given dataset.')
    
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
    return parser.parse_args()

def make_path(data_dir, base, exp_name):
    train_path = os.path.join(data_dir, 'Train')
    valid_path = os.path.join(data_dir, 'Validation')
    
    if (not os.path.isdir(train_path)) or (not os.path.isdir(valid_path)):
        print('Please split images into train directory and validation directory.')
        exit()
        
    freeze_weight_save_path = os.path.join(base, 'train/save_plots_initial/{exp}/{model}.h5'.format(exp = exp_name, model = model_name))
    unfreeze_weight_save_path = os.path.join(base, 'train/save_plots_initial/{exp}/{model}.h5'.format(exp = exp_name, model = model_name))
    freeze_img_save_path = os.path.join(base, 'train/save_plots_initial/{}'.format(exp_name))
    unfreeze_img_save_path = os.path.join(base, 'train/save_plots_final/{}'.format(exp_name))
    
    if not os.path.exists(os.path.join(base, 'train/save_plots_initial/{}/'.format(exp_name))):
        os.makedirs(os.path.join(base, 'train/save_plots_initial/{}/'.format(exp_name)))
    
    if not os.path.exists(os.path.join(base, 'train/save_plots_initial/{}/'.format(exp_name))):
        os.makedirs(os.path.join(base, 'train/save_plots_initial/{}/'.format(exp_name)))
        
    if not os.path.exists(freeze_img_save_path):
        os.makedirs(freeze_img_save_path)
        
    if not os.path.exists(unfreeze_img_save_path):
        os.makedirs(unfreeze_img_save_path)
    
    return train_path, valid_path, freeze_weight_save_path, unfreeze_weight_save_path, freeze_img_save_path, unfreeze_img_save_path

if __name__=='__main__':
    
    batch_size = 16
    rotation_range = 20
    height_shift = 0.05
    width_shift = 0.05
    
    args = get_args()

    data_path = args.path[0]
    model_name = args.model_name[0]
    img_size = args.img_size[0]
    
    nih_weight = 'nih_weights_{name}.h5'.format(name = model_name)

    if not os.path.exists(nih_weight):
        print('NIH weight does not exists.'
              ' Please provide a NIH weight file in the format of nih_weight_[model name].h5')
        exit()
    
    exp_name = 'train_individual'
    base_dir = os.getcwd()
    
    train_dir, valid_dir, weight_dir1, weight_dir2, img_dir1, img_dir2 = make_path(data_path, base_dir, exp_name)

    img_proc = imgUtils(img_size)
    train_idg, val_idg = img_proc.dataGen(rotation_range, height_shift, width_shift)
    train_gen, val_gen = img_proc.generator(batch_size, train_idg, val_idg, train_dir, valid_dir)

    lr = 0.001
    momentum = 0.9
    nestrov = True
    dropout_rate = 0.3
    
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
    
    dropout_model = features.getDropoutModel(model_name, img_size, nih_weight, dropout_rate)
    features.compileModel(dropout_model, lr, momentum, nestrov)
    model_history = features.generator(dropout_model, train_gen, val_gen, pre_epoch, cp, rlr, es)
    img_proc.plot_save(model_history, img_dir1, exp_name)

    # Unfreeze and train the entier model
    model = features.load(dropout_model, weight_dir1)
    model = features.unfreeze(model)

    patience_es = 10
    
    es = features.setES(monitor, patience_es, min_delta)
    cp = features.setCP(monitor, weight_dir2)
    
    features.compileModel(model, lr, momentum, nestrov)
    model_history = features.generator(dropout_model, train_gen, val_gen, epoch, cp, rlr, es)
    img_proc.plot_save(model_history, img_dir2, exp_name)
    
    

    



# Pretrain a given model on NIH dataset
import argparse
import os
from utils import nihUtils, imgUtils, trainFeatures

def get_args():
    # Implement command line argument
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
                        help='the path that contains NIH dataset and NIH csv file or the path in which a new directory for NIH dataset will be created.')
    return parser.parse_args()

if __name__=='__main__':

    args = get_args()
    nih_path = args.path[0]
    img_size = args.img_size
    model_name = args.model_name

    batch_size = 16
    rotation_range = 20
    height_shift = 0.05
    width_shift = 0.05

    nih = nihUtils()
    nih_path, create_dir = nih.createDir(nih_path)
    model_save_path = 'nih_weight_{name}.h5'.format(name = model_name)
    
    if create_dir:
        nih.nihDownload(nih_path)
        
    nih_img_path = nih_path
    nih_csv_path = nih_path
    csv_name = os.path.join(nih_csv_path, 'NIH_Data_Entry.csv')

    train_df, val_df, labels = nih.nihSplit(csv_name, nih_img_path)
    
    img_proc = imgUtils(img_size)
    train_idg, val_idg = img_proc.dataGen(rotation_range, height_shift, width_shift)
    
    train_generator, val_generator = nih.nihGenerator(img_size, 
                                                      batch_size, train_idg, val_idg, 
                                                      train_df, val_df, labels)

    
    # Train a given model on NIH dataset
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
           
    _, model, _ = features.getModel(model_name, 'imagenet')
    features.compileModel(model)
    
    epochs = 50
    features.NIHgenerator(model, batch_size, train_generator, val_generator, epochs, cp, rlp, es)







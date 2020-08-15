# Pretrain a given model on NIH dataset
import argparse
import os
from utils import imgUtils, trainFeatures
from covid_models import hyperModel
import kerastuner
from kerastuner.tuners import BayesianOptimization

def get_args():
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
    return parser.parse_args()

def make_path(data_dir, base, exp_name):
    train_path = os.path.join(data_dir, 'Train')
    valid_path = os.path.join(data_dir, 'Validation')
    
    if (not os.path.isdir(train_path)) or (not os.path.isdir(valid_path)):
        print('Please split images into train directory and validation directory.')
        exit()
    
    if not os.path.isdir(os.path.join(base, 'tuner')):
        os.mkdir(os.path.join(base, 'tuner'))
        
    freeze_save_path = os.path.join(base, '/tuner/initial_cps/{}.h5'.format(model_name))
    unfreeze_save_path = os.path.join(base, '/tuner/cps/{}.h5'.format(model_name))
    best_model_path = os.path.join(base, '/tuner/best_models_and_params/{}/model'.format(exp_name))
    best_weight_path = os.path.join(base, '/tuner/best_models_and_params/{}/model_weights.h5'.format(exp_name))
    best_param_path = os.path.join(base, '/tuner/best_models_and_params/{}/model_params'.format(exp_name))
    
    if not os.path.exists(os.path.join(base, '/tuner/initial_cps/'.format(exp_name))):
        os.makedirs(os.path.join(base, '/tuner/initial_cps/{}'.format(exp_name)))
        
    if not os.path.exists(os.path.join(base, '/tuner/cps/{}'.format(exp_name))):
        os.makedirs(os.path.join(base, '/tuner/cps/{}'.format(exp_name)))
    
    if not os.path.exists(os.path.join(base, '/tuner/best_models_and_params/{}/model'.format(exp_name))):
        os.makedirs(os.path.join(base, '/tuner/best_models_and_params/{}/model'.format(exp_name)))
    
    return train_path, valid_path, freeze_save_path, unfreeze_save_path, best_model_path, best_weight_path, best_param_path
        
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
    
    exp_name = 'tuner_pretrain'
    base_dir = os.getcwd()
    
    train_dir, valid_dir, freeze_dir, unfreeze_dir, model_dir, weight_dir, param_dir = make_path(data_path, base_dir, exp_name)

    img_proc = imgUtils(img_size)
    train_idg, val_idg = img_proc.dataGen(rotation_range, height_shift, width_shift)
    train_gen, val_gen = img_proc.generator(batch_size, train_idg, val_idg, train_dir, valid_dir)

    lr = 0.001
    momentum = 0.9
    nestrov = True
    patience_rlr = 3
    patience_es = 5
    factor = 0.1
    min_delta = 0.001
    monitor = 'val_auc'
    epoch = 50

    features = trainFeatures()
    rlr = features.setRLP(monitor, factor, patience_rlr)
    es = features.setES(monitor, patience_es, min_delta)
    cp = features.setCP(monitor, freeze_dir)
    
    freeze_model, model, base = features.getModel(model_name, img_size, nih_weight)
    features.compileModel(freeze_model, lr, momentum, nestrov)

    model_history = features.generator(freeze_model, train_gen, val_gen, epoch, cp, rlr, es)
    img_proc.plot_save(model_history, base_dir, exp_name)

    # Add dropout layer and run tuner with entire model
    
    model = features.load(model, freeze_dir)
    model = features.unfreeze(freeze_model)

    patience_es = 10
    
    es = features.setES(monitor, patience_es, min_delta)
    cp = features.setCP(monitor, unfreeze_dir)

    hp = hyperModel(base, freeze_dir)

    TOTAL_TRIALS = 10
    EXECUTION_PER_TRIAL = 1
    EPOCHS = 50

    tuner = BayesianOptimization(
        hp,
        max_trials=TOTAL_TRIALS,
        objective=kerastuner.Objective("val_auc", direction="max"),
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory='bayesian',
        project_name=exp_name
    )

    history = tuner.search(train_gen, 
                           epochs=EPOCHS,
                           validation_data=val_gen, 
                            callbacks = [es, cp],
                            verbose =2,
                            use_multiprocessing=False)

    # Save best model and weight
    best_model = tuner.get_best_models(num_models=TOTAL_TRIALS)[0]
    best_config = best_model.optimizer.get_config()
    
    hyperparameters = tuner.get_best_hyperparameters(num_trials=TOTAL_TRIALS)
    best_hyperparameters = hyperparameters[0].get_config()
    
    model.save(model_dir)
    model.save_weights(weight_dir)
    with open(param_dir, "w") as text_file:
        text_file.write(str(best_hyperparameters))




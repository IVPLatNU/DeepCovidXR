# Pretrain a given model on NIH dataset
import argparse
import os
from utils import imgUtils, trainFeatures
from covid_models import hyperModel, DenseNet, ResNet, XceptionNet, EfficientNet, InceptionNet, InceptionResNet 
import kerastuner
from kerastuner.tuners import BayesianOptimization

# Implement command line argument
parser = argparse.ArgumentParser(description='Use keras tuner to find best hyper parameter.')
parser.add_argument('-m', '--model', dest = 'model_name')

parser.add_argument('model', metavar = 'model_name', 
                    choices = ['ResNet-50', 'Xception', 'DenseNet-121', 'Inception-V3', 
                               'Inception-ResNet-V2', 'EfficientNet-B2'],
                    type = str, nargs = 1,
                    help = 'the name of the model to be trained.\n Choose from ResNet-50, Xception, DenseNet-121, Inception-V3,' 
                            'Inception-ResNet-V2, EfficientNet-B2')

parser.add_argument('-s', '--size', dest = 'img_size')

parser.add_argument('size', metavar = 'img_size',
                    type = int, nargs = 1,
                    help = 'the size of dataset images')

parser.add_argument('path', metavar='DATA_path', type=str, nargs=1,
                    help='the path that contains the dataset.')

args = parser.parse_args()
data_path = args.path[0]
model_name = args.model_name
img_size = args.img_size

train_path = data_path + '\\Train'
valid_path = data_path + '\\Validation'
nih_weight = 'nih_weights_{name}.h5'.format(name = model_name)

if not os.path.exists(nih_weight):
    print('NIH weight does not exists.'
          ' Please provide a NIH weight file in the format of nih_weight_[model name].h5')
    exit()
    
exp_name = 'tuner_pretrain'
base_folder = os.getcwd()

if not os.path.exists(base_folder + 'initial_cps/{}'.format(exp_name)):
    os.makedirs(base_folder + 'initial_cps/{}'.format(exp_name))
    
if not os.path.exists(base_folder + 'cps/{}'.format(exp_name)):
    os.makedirs(base_folder + 'cps/{}'.format(exp_name))

if not os.path.exists(base_folder + 'best_models_and_params/{}/model'.format(exp_name)):
    os.makedirs(base_folder + 'best_models_and_params/{}/model'.format(exp_name))

freeze_save_path = base_folder + 'initial_cps/{}.h5'.format(model_name)
unfreeze_save_path = base_folder + 'cps/{}'.format(exp_name)
best_model_path = base_folder + 'best_models_and_params/{}/model'.format(exp_name)
best_weight_path = base_folder + 'best_models_and_params/{}/model_weights.h5'.format(exp_name)
best_param_path = base_folder + 'best_models_and_params/{}/model_params'.format(exp_name)

batch_size = 16
rotation_range = 20
height_shift = 0.05
width_shift = 0.05

img_proc = imgUtils(img_size)
train_idg, val_idg = img_proc.dataGen(rotation_range, height_shift, width_shift)
train_gen, val_gen = img_proc.generator(batch_size, train_idg, val_idg, train_path, valid_path)

lr = 0.001
momentum = 0.9
nestrov = True

# Train with only pooling layers

if model_name == 'ResNet-50':
    resnet = ResNet(nih_weight)
    base = resnet.buildTunerModel(img_size)
    model = resnet.buildBaseModel(img_size)
    model = resnet.freeze(model)
    resnet.compileModel(model, lr, momentum, nestrov)
    
elif model_name == 'Xception':
    xception = XceptionNet(nih_weight)
    base = xception.buildTunerModel(img_size)
    model = xception.buildBaseModel(img_size)
    model = xception.freeze(model)
    xception.compileModel(model, lr, momentum, nestrov)

elif model_name == 'DenseNet-121':
    dense = DenseNet(nih_weight)
    base = dense.buildTunerModel(img_size)
    model = dense.buildBaseModel(img_size)
    model = dense.freeze(model)
    dense.compileModel(model, lr, momentum, nestrov)
    
elif model_name == 'Inception-V3':
    inception = InceptionNet(nih_weight)
    base = inception.buildTunerModel(img_size)
    model = inception.buildBaseModel(img_size)
    model = inception.freeze(model)
    inception.compileModel(model, lr, momentum, nestrov)
    
elif model_name == 'Inception-ResNet-V2':
    inceptionres = InceptionResNet(nih_weight)
    base = inceptionres.buildTunerModel(img_size)
    model = inceptionres.buildBaseModel(img_size)
    model = inceptionres.freeze(model)
    inceptionres.compileModel(model, lr, momentum, nestrov)
    
elif model_name == 'EfficientNet-B2':
    efficient = EfficientNet(nih_weight)
    base = efficient.buildTunerModel(img_size)
    model = efficient.buildBaseModel(img_size)
    model = efficient.freeze(model)
    efficient.compileModel(model, lr, momentum, nestrov)

patience_rlr = 3
patience_es = 5
factor = 0.1
min_delta = 0.001
monitor = 'val_auc'
epoch = 50

features = trainFeatures()
rlr = features.setRLP(monitor, factor, patience_rlr)
es = features.setES(monitor, patience_es, min_delta)
cp = features.setCP(monitor, freeze_save_path)

model_history = features.generator(model, train_gen, val_gen, epoch, cp, rlr, es)
img_proc.plot_save(model_history, base_folder, exp_name)

# Add dropout layer and run tuner with entire model

model = features.load(model, freeze_save_path)
model = features.unfreeze(model)

patience_es = 10

es = features.setES(monitor, patience_es, min_delta)
cp = features.setCP(monitor, unfreeze_save_path)

hp = hyperModel(base, freeze_save_path)

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

model.save(best_model_path)
model.save_weights(best_weight_path)
with open(best_param_path, "w") as text_file:
    text_file.write(str(best_hyperparameters))




# Ensemble trained models and generates confusion matrices for 224 and 331 images

import argparse
import os
from utils import imgUtils, trainFeatures
import numpy as np
from deepstack.ensemble import StackEnsemble
from sklearn.ensemble import RandomForestRegressor
from deepstack.ensemble import DirichletEnsemble
from deepstack.base import KerasMember
from tqdm import tqdm

batch_size = 100
rotation_range = 20
height_shift = 0.05
width_shift = 0.05

def get_args():
    # Implement command line argument
    parser = argparse.ArgumentParser(description='Ensemble trained models to generate confusion matrices.')
    parser.add_argument('--weight', '-w', dest='weight_path', metavar='weight_path', 
                        type=str, nargs=1,
                        required = True, help='the path that contains trained weights.')
    
    parser.add_argument('--data', '-d', dest='data_path', 
                        metavar='CROPPED_DATA_path', type=str, nargs=1,
                        required = True, help='the path that contains the entire dataset.')
    
    return parser.parse_args()

def get_generator(data_path, batch_size):
    all_ntta_generators = []
    all_tta_generators = []
    dir_list = []
    crop_224_valid_dir = os.path.join(data_path, '224\crop\Validation')
    crop_224_test_dir = os.path.join(data_path, '224\crop\Test')
    uncrop_224_valid_dir = os.path.join(data_path, '224\\uncrop\Validation')
    uncrop_224_test_dir = os.path.join(data_path, '224\\uncrop\Test')
    crop_331_valid_dir = os.path.join(data_path, '331\crop\Validation')
    crop_331_test_dir = os.path.join(data_path, '331\\crop\Test')
    uncrop_331_valid_dir = os.path.join(data_path, '331\\uncrop\Validation')
    uncrop_331_test_dir = os.path.join(data_path, '331\\uncrop\Test')
    
    dir_list.extend([crop_224_valid_dir, uncrop_224_valid_dir, 
                     crop_224_test_dir, uncrop_224_test_dir,
                     crop_331_valid_dir, uncrop_331_valid_dir,
                     crop_331_test_dir, uncrop_331_test_dir])
    
    if not (os.path.exists(crop_224_valid_dir) and os.path.exists(crop_331_valid_dir)
            and os.path.exists(uncrop_224_valid_dir) and os.path.exists(uncrop_331_valid_dir) 
            and os.path.exists(crop_224_test_dir) and os.path.exists(uncrop_224_test_dir) 
            and os.path.exists(crop_331_test_dir) and os.path.exists(uncrop_331_test_dir) ):
        print('Data path is invalid. Please check if data path contains directory for 224 and 331,'
              'cropped and uncropped data. ')
        exit()
        
    img_proc1 = imgUtils(224)
    
    train_idg_224, val_idg_224 = img_proc1.dataGen(rotation_range, height_shift, width_shift)
    
    for i in range(4):
        test_gen = img_proc1.testGenerator(batch_size, val_idg_224, dir_list[i])
        if i%2 == 1:
            train_gen = img_proc1.testGenerator(batch_size, train_idg_224, dir_list[i])
            all_tta_generators.append(train_gen)
        all_ntta_generators.append(test_gen)
        
    
    img_proc2 = imgUtils(331)
    
    train_idg_331, val_idg_331 = img_proc1.dataGen(rotation_range, height_shift, width_shift)
    
    for i in range(4):
        test_gen = img_proc2.testGenerator(batch_size, val_idg_331, dir_list[i+4])
        if i%2 == 1:
            train_gen = img_proc1.trainGenerator(batch_size, train_idg_331, dir_list[i+4])
            all_tta_generators.append(train_gen)
        all_ntta_generators.append(test_gen)
        
    for i in range(3):
        for j in range(4):
            all_ntta_generators.append(all_tta_generators[j])
        
    return all_ntta_generators, all_tta_generators

def create_member(model_name, model, generator_list):
    name_parts = model_name.split("_")
    if "224" in name_parts and "crop" in name_parts:
        member = KerasMember(name = model_name, keras_model = model, 
                             train_batches = generator_list[0], val_batches = generator_list[2])
    elif "224" in name_parts and "uncrop" in name_parts:
         member = KerasMember(name = model_name, keras_model = model, 
                             train_batches = generator_list[1], val_batches = generator_list[3])
    elif "331" in name_parts and "crop" in name_parts:
         member = KerasMember(name = model_name, keras_model = model, 
                             train_batches = generator_list[4], val_batches = generator_list[6])
    elif "331" in name_parts and "uncrop" in name_parts:
        member = KerasMember(name = model_name, keras_model = model, 
                             train_batches = generator_list[5], val_batches = generator_list[7])
    
    return member
    

def get_members(ntta_generator_list, weight_path):
    model_list = []
    model_name_list = []
    member_list = []
    features = trainFeatures()
    res_224_crop, xception_224_crop, dense_224_crop, inception_224_crop, inceptionres_224_crop, efficient_224_crop = features.getAllModel(img_size1, weight_path, 'crop')
    model_list.extend([res_224_crop, xception_224_crop, dense_224_crop, 
                       inception_224_crop, inceptionres_224_crop, efficient_224_crop])
    model_name_list.extend(["res_224_crop", "xception_224_crop", "dense_224_crop", 
                       "inception_224_crop", "inceptionres_224_crop", "efficient_224_crop"])
    
    res_224_uncrop, xception_224_uncrop, dense_224_uncrop, inception_224_uncrop, inceptionres_224_uncrop, efficient_224_uncrop = features.getAllModel(img_size1, weight_path, 'uncrop')
    model_list.extend([res_224_uncrop, xception_224_uncrop, dense_224_uncrop, 
                       inception_224_uncrop, inceptionres_224_uncrop, efficient_224_uncrop])
    model_name_list.extend(["res_224_uncrop", "xception_224_uncrop", "dense_224_uncrop", 
                       "inception_224_uncrop", "inceptionres_224_uncrop", "efficient_224_uncrop"])
    
    res_331_crop, xception_331_crop, dense_331_crop, inception_331_crop, inceptionres_331_crop, efficient_331_crop = features.getAllModel(img_size2, weight_path, 'crop')
    model_list.extend([res_331_crop, xception_331_crop, dense_331_crop, 
                       inception_331_crop, inceptionres_331_crop, efficient_331_crop])
    model_name_list.extend(["res_331_crop", "xception_331_crop", "dense_331_crop", 
                       "inception_331_crop", "inceptionres_331_crop", "efficient_331_crop"])
    
    res_331_uncrop, xception_331_uncrop, dense_331_uncrop, inception_331_uncrop, inceptionres_331_uncrop, efficient_331_uncrop = features.getAllModel(img_size2, weight_path, 'uncrop')
    model_list.extend([res_331_uncrop, xception_331_uncrop, dense_331_uncrop, 
                       inception_331_uncrop, inceptionres_331_uncrop, efficient_331_uncrop])
    model_name_list.extend(["res_331_uncrop", "xception_331_uncrop", "dense_331_uncrop", 
                       "inception_331_uncrop", "inceptionres_331_uncrop", "efficient_331_uncrop"])
    
    for i in range(24):
        member = create_member(model_name_list[i], model_list[i], ntta_generator_list)
        member_list.append(member)
        
    return member_list, model_list, model_name_list

def ensemble_members(member_list, model_list, tta_generator_list):
    wAvgEnsemble = DirichletEnsemble()
    wAvgEnsemble.add_members(member_list)
    wAvgEnsemble.fit()
    wAvgEnsemble.describe()
        
    stack = StackEnsemble()
    stack.model = RandomForestRegressor(verbose=1, n_estimators=200, 
                                      max_depth=50, min_samples_split=20)
    stack.add_members(member_list)
    stack.fit()
    stack.describe()
    
    combined_weighted_probs_notta = []
    combined_probs_notta = []
    
    # Predictions without test time augmentation
    for member, weight in zip(member_list, wAvgEnsemble.bestweights):
        weighted_probs = np.multiply(member.val_probs, weight)
        combined_weighted_probs_notta.append(weighted_probs)
        combined_probs_notta.append(member.val_probs)
    
    combined_weighted_probs_notta = np.asarray(combined_weighted_probs_notta)
    combined_probs_notta = np.asarray(combined_probs_notta)
    ensemble_pred_notta = np.sum(combined_weighted_probs_notta)
    ensemble_pred_round_notta = np.round(ensemble_pred_notta)
    
    # Predictions with test time augmentation
    tta_steps = 10
    combined_weighted_probs = []
    combined_probs = []
    
    for model, data_generator, weight in zip(model_list, tta_generator_list, wAvgEnsemble.bestweights):
        predictions = []
        for i in tqdm(range(tta_steps)):
            preds = model.predict(data_generator, verbose=1)
            predictions.append(preds)
        Y_pred = np.mean(predictions, axis=0)
        y_pred = np.multiply(Y_pred, weight)
        combined_probs.append(Y_pred)
        combined_weighted_probs.append(y_pred)
    
    combined_probs = np.asarray(combined_probs)
    combined_weighted_probs = np.asarray(combined_weighted_probs)
    
    ensemble_pred  = np.sum(combined_weighted_probs, axis=0)
    ensemble_pred_round = np.round(ensemble_pred)

    cm_generator_224 = tta_generator_list[3]
    cm_generator_331 = tta_generator_list[7]
    
    proc1 = imgUtils(224)
    proc2 = imgUtils(331)
    
    cm_224 = proc1.confusionMatrix(cm_generator_224, ensemble_pred_round)
    cm_331 = proc2.comfusionMatrix(cm_generator_331, ensemble_pred_round)
    
    return cm_224, cm_331

if __name__=='__main__':
    
    args = get_args()
    weights = args.weight_path[0]
    data_dir = args.data_path[0]

    img_size1 = 224
    img_size2 = 331
     
    ntta_generators, tta_generators, cm_generators = get_generator(data_dir, batch_size)
    member_list, model_list, model_name_list = get_members(ntta_generators, weights)
    cm_224, cm_331 = ensemble_members(member_list, model_list, tta_generators)

    
    
    

    





# Ensemble trained models and produce weights for each model prediction

import argparse
import os
from utils import imgUtils
import numpy as np
from deepstack.ensemble import DirichletEnsemble
from deepstack.base import KerasMember
from evaluate import get_datagenerators_folders
from predict import get_model_list
import pandas as pd
import pickle

def get_args():
    """
    This function gets various user input from command line. The user input variables
    include the path to pretrained weight files, the path to the dataset and the path
    where the output should be saved.
    
    Returns:
        parser.parse_args() (list): a list of user input values.

    """
    # Implement command line argument
    parser = argparse.ArgumentParser(description='Ensemble trained models to generate confusion matrices.')
    parser.add_argument('--weight', '-w', dest='weight_path', metavar='weight_path',
                        type=str, nargs=1,
                        required=True, help='the path that contains trained weights.')

    parser.add_argument('--data', '-d', dest='data_path',
                        metavar='DATA_path', type=str, nargs=1,
                        required=True, help='the path that contains the entire dataset.')

    parser.add_argument('--output', '-o', dest='output', metavar='prediction_output_path', type=str,
                        default=None, help='the directory to output a csv file of predictions and pickled list of '
                                           'ensemble weights; if not provided will save to current working directory')

    return parser.parse_args()


def get_generator(data_path):
    """
    This function creates a list of validation and test image generators. 
    
    Parameters:
        data_path (string): the path to the parent directory of the images.
    
    Returns:
        combined_gen (list): a cmobined list of validation and test image generators.
        The first four members are validation generators for uncropped 224x224 images,
        cropped 224x224 images, uncropped 331x331 images and cropped 331x331 images.
        The last four members are test generators for images in the same order as 
        the first four members. All generators have batch size 16.
        To learn more about the structure of the directories, check out Readme.md.

    """

    crop_224_valid_dir = os.path.join(data_path, '224', 'crop', 'Validation')
    crop_224_test_dir = os.path.join(data_path, '224', 'crop', 'Test')
    uncrop_224_valid_dir = os.path.join(data_path, '224', 'uncrop', 'Validation')
    uncrop_224_test_dir = os.path.join(data_path, '224', 'uncrop', 'Test')
    crop_331_valid_dir = os.path.join(data_path, '331', 'crop', 'Validation')
    crop_331_test_dir = os.path.join(data_path, '331', 'crop', 'Test')
    uncrop_331_valid_dir = os.path.join(data_path, '331', 'uncrop', 'Validation')
    uncrop_331_test_dir = os.path.join(data_path, '331', 'uncrop', 'Test')

    if not (os.path.exists(crop_224_valid_dir) and os.path.exists(crop_331_valid_dir)
            and os.path.exists(uncrop_224_valid_dir) and os.path.exists(uncrop_331_valid_dir)
            and os.path.exists(crop_224_test_dir) and os.path.exists(uncrop_224_test_dir)
            and os.path.exists(crop_331_test_dir) and os.path.exists(uncrop_331_test_dir)):
        print('Data path is invalid. Please check that directory tree is set up as described in README file.')
        exit()

    valid_gen = get_datagenerators_folders(uncrop_224_valid_dir, crop_224_valid_dir, uncrop_331_valid_dir,
                                           crop_331_valid_dir, batch_size=16)

    test_gen = get_datagenerators_folders(uncrop_224_test_dir, crop_224_test_dir, uncrop_331_test_dir,
                                          crop_331_test_dir, batch_size=16)

    # test_gen_tta = get_datagenerators_folders(uncrop_224_test_dir, crop_224_test_dir, uncrop_331_test_dir,
    #                                           crop_331_test_dir, tta=True)

    combined_gen = valid_gen[0:4] + test_gen[0:4]

    return combined_gen


def create_member(model_name, model, generator_list):
    #'''Create a member of model ensemble'''
    """
    This function creates a member of model ensemble. 
    
    Parameters:
        model_name (string): the name of the model. Must be a member of the model_name_list
        as provided in the function get_members.
        model (class): a keras model corresponds to the model name.
        generator_list (list): a list of test and validation image generators.
        
    
    Returns:
        member (class): a member object for model ensembling.

    """

    name_parts = model_name.split("_")
    if "224" in name_parts and "uncrop" in name_parts:
        member = KerasMember(name=model_name, keras_model=model,
                             train_batches=generator_list[0], val_batches=generator_list[4])
    elif "224" in name_parts and "crop" in name_parts:
        member = KerasMember(name=model_name, keras_model=model,
                             train_batches=generator_list[1], val_batches=generator_list[5])
    elif "331" in name_parts and "uncrop" in name_parts:
        member = KerasMember(name=model_name, keras_model=model,
                             train_batches=generator_list[2], val_batches=generator_list[6])
    elif "331" in name_parts and "crop" in name_parts:
        member = KerasMember(name=model_name, keras_model=model,
                             train_batches=generator_list[3], val_batches=generator_list[7])

    return member


def get_members(combined_generator_list, weight_path):
    #'''Creates the list of members for ensembling from a list of data generators and corresponding model weights'''
    """
    This function creates a list of ensembling memebers from data generators and
    correspongding model weights. 
    
    Parameters:
        combined_generator_list (list): a list of validation and test image data generators.
        weight_path (string): the path to pretrained weight files.
        
    
    Returns:
        member_list (list): a list of model members for ensembling.

    """
    
    model_list = get_model_list(weight_path)

    model_name_list = ['dense_224_uncrop',
                       'dense_224_crop',
                       'dense_331_uncrop',
                       'dense_331_crop',
                       'res_224_uncrop',
                       'res_224_crop',
                       'res_331_uncrop',
                       'res_331_crop',
                       'inception_224_uncrop',
                       'inception_224_crop',
                       'inception_331_uncrop',
                       'inception_331_crop',
                       'inceptionresnet_224_uncrop',
                       'inceptionresnet_224_crop',
                       'inceptionresnet_331_uncrop',
                       'inceptionresnet_331_crop',
                       'xception_224_uncrop',
                       'xception_224_crop',
                       'xception_331_uncrop',
                       'xception_331_crop',
                       'efficient_224_uncrop',
                       'efficient_224_crop',
                       'efficient_331_uncrop',
                       'efficient_331_crop']

    member_list = []

    for model_name, model in zip(model_name_list, model_list):
        member = create_member(model_name, model, combined_generator_list)
        member_list.append(member)

    return member_list


def ensemble_members(member_list):
    """
    This function calculates weights for each model of an ensemble for weighted 
    averaging of predictions using random search of a Dirichlet distribution. 
    
    Parameters:
        member_list (list): a list of model members for ensembling.
        
    
    Returns:
        wAvgEnsemble.bestweights (list): a list of float numbers which represents
        the best weight for each ensembled model.
        
        ensemble_pred (float): the weighted sum of each ensembled member's predicted
        probability for a single image.
        
        ensemble_pred_round (int): the rounded weighted sum of each emsembled 
        member's predicted probability for a single image. Can either be 1 or 0.
        
        individual_preds (pandas dataframe): the predicted probability of each 
        ensembled model for a single image.

    """
    #'''Calculates weights for each model of an ensemble for weighted averaging of predictions using random
    #search of a Dirichlet distribution'''
    wAvgEnsemble = DirichletEnsemble()
    wAvgEnsemble.add_members(member_list)
    wAvgEnsemble.fit()
    wAvgEnsemble.describe()

    combined_weighted_probs = []
    combined_probs = []

    # Predictions without test time augmentation
    for member, weight in zip(member_list, wAvgEnsemble.bestweights):
        weighted_probs = np.multiply(member.val_probs, weight)
        combined_weighted_probs.append(weighted_probs)
        combined_probs.append(member.val_probs)

    # combined_probs = np.asarray(combined_probs)
    combined_weighted_probs = np.asarray(combined_weighted_probs)

    individual_preds = pd.DataFrame(np.squeeze(np.stack(combined_probs, axis=-1)), columns = [member.name for member in member_list])

    ensemble_pred = np.sum(combined_weighted_probs, axis=0)
    ensemble_pred_round = np.round(ensemble_pred)

    return wAvgEnsemble.bestweights, ensemble_pred, ensemble_pred_round, individual_preds


if __name__ == '__main__':
    """
    The main function ensembles individual models and store the trained results
    for each model and each image to "individual_predictions.csv". The weighted 
    sum of the predicted probability are saved to "ensemble_prediction.csv". 

    """
    args = get_args()
    weights = os.path.normpath(args.weight_path[0])
    data_dir = os.path.normpath(args.data_path[0])
    output_path = args.output
    if output_path is not None:
        output_path = os.path.normpath(output_path)
    else:
        output_path = os.getcwd()

    #Calculate model weights for weighted average ensemble
    print('Ensembling models...')
    combined_gen = get_generator(data_dir)
    member_list = get_members(combined_gen, weights)
    print(member_list)
    print(len(member_list))
    weights, ensemble_pred, ensemble_pred_round, results_individual = ensemble_members(member_list)

    #Save results of weighted average ensemble
    print('Saving results...')
    files = combined_gen[0].filenames
    actual = combined_gen[0].classes
    scores = ensemble_pred.flatten()
    binary_pred = ensemble_pred_round.flatten()
    results = pd.DataFrame.from_dict({'Filenames': files,
                                      'Scores': scores,
                                      'Binary prediction': binary_pred,
                                      'Actual': actual})
    results_individual.insert(0, 'filenames', files)
    results_path = os.path.join(output_path, 'Ensemble_Results')
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    results.to_csv(os.path.join(results_path, 'ensemble_predictions.csv'))
    results_individual.to_csv(os.path.join(results_path, 'individual_predictions.csv'))
    pickle.dump(weights, open(os.path.join(results_path, 'ensemble_weights.pickle'), 'wb'))
    imgUtils.confusionMatrix(actual, binary_pred, results_path)
    imgUtils.ROCcurve(actual, scores, results_path)
    print('Done')

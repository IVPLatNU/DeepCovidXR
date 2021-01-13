# Test individual image
import argparse
from utils import imgUtils, trainFeatures
import numpy as np
import pickle
from tqdm import tqdm
import os
import pandas as pd
from resize_img import resize_images
import glob
import gc
import tensorflow as tf
from crop_img import lungseg_one_process
import pathlib

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

"""Image probability prediction

This script will predict the probability of COVID-19 positive for a image test set.

The result will be predicted with a trained ensembled model. The image can be optionally
resized to 224x224 and 331x331. To modify this, check more details in the main function.
"""

def get_args():
    """
    This function gets various user input form command line. The user input variables
    include the path to pretrained weight files, the path to the dataset, the path
    where the output should be saved, the test-time augmentation switch and the 
    path to the ensemble weights as a pickled list.
    
    Returns:
        parser.parse_args() (list): a list of user input values.
    """
    
    # Implement command line argument
    parser = argparse.ArgumentParser(
        description='For each input image, generates predictions of COVID-19 status.')
    parser.add_argument('--weight', '-w', dest='weight_path', metavar='weight_path',
                        type=str,
                        required=True, help='the path that contains trained weights.')

    parser.add_argument('--image', '-i', dest='img_path',
                        metavar='IMAGE_path', type=str,
                        required=True, help='the path to the image/folder of images.')

    parser.add_argument('--ensemble_weight', '-e', dest='ensemble_weights',
                        metavar='ensemble_weight_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'ensemble_weights.pickle'),
                        required=False, help='the path to the ensemble weights as a pickled list, '
                                             'if not supplied uses our pretrained weights')

    parser.add_argument('--tta', '-t', action='store_true', dest='tta',
                        help='switch to turn on test-time augmentation, warning: this takes significantly longer as '
                             'each model prediction is run 10 times')

    parser.add_argument('--output', '-o', dest='output', metavar='prediction_output_path', type=str,
                        default=None, help='the directory to output a csv file of predictions, if not provided '
                                             'predictions will not be saved, only printed in the terminal')

    return parser.parse_args()

def mkdirs(img_dir):
    """
    This function creates directories for different types of image augmentation.
    
    Parameters:
        img_dir (string): the path to the dataset parent directory.
    
    Returns:
        dir_224/331_crop/uncrop (string): the new paths created for different image
        augmentations.
    """
    
    img_dir = os.path.dirname(img_dir)

    dir_224 = os.path.join(img_dir, '224')
    if not os.path.isdir(dir_224):
        os.mkdir(dir_224)

    dir_224_uncrop = os.path.join(dir_224, 'uncrop')
    if not os.path.isdir(dir_224_uncrop):
        os.mkdir(dir_224_uncrop)

    dir_224_crop = os.path.join(dir_224, 'crop')
    if not os.path.isdir(dir_224_crop):
        os.mkdir(dir_224_crop)

    dir_331 = os.path.join(img_dir, '331')
    if not os.path.isdir(dir_331):
        os.mkdir(dir_331)

    dir_331_uncrop = os.path.join(dir_331, 'uncrop')
    if not os.path.isdir(dir_331_uncrop):
        os.mkdir(dir_331_uncrop)

    dir_331_crop = os.path.join(dir_331, 'crop')
    if not os.path.isdir(dir_331_crop):
        os.mkdir(dir_331_crop)

    return dir_224_uncrop, dir_224_crop, dir_331_uncrop, dir_331_crop


def get_crop(input_path):
    """
    This function crops out lung field for a given set of x-ray images.
    
    Parameters:
        input_path (string): the path to the x-ray image dataset to be cropped.
    
    Returns:
        output_dir (string): the path to the cropped image dataset.
        augmentations.
    """
    
    current_path = os.path.dirname(__file__)
    unet_path = pathlib.Path(os.path.join(current_path, 'trained_unet_model.hdf5'))
    output_dir = pathlib.Path(os.path.dirname(input_path))

    if not os.path.isdir(input_path):
        lungseg_one_process(output_dir, unet_path, filenames=[pathlib.Path(input_path)], out_pad_size=8, debug=False)
    else:
        filenames = glob.glob(os.path.join(input_path, '*'))
        filenames = [pathlib.Path(filename) for filename in filenames]
        lungseg_one_process(output_dir, unet_path, filenames=filenames, out_pad_size=8, debug=False)
    print('Completed!')
    return output_dir


def get_model_list(weight_path):
    """
    This function creates a list of models used in ensembling.
    
    Parameters:
        weight_path (string): the path to the directory with pretrained weights.
    
    Returns:
        model_list (list): a list of keras models used in ensembling.
    """
    
    features = trainFeatures()
    res_224_crop, xception_224_crop, dense_224_crop, inception_224_crop, inceptionresnet_224_crop, efficient_224_crop = features.getAllModelFast(
        224, weight_path, 'crop')

    res_224_uncrop, xception_224_uncrop, dense_224_uncrop, inception_224_uncrop, inceptionresnet_224_uncrop, efficient_224_uncrop = features.getAllModelFast(
        224, weight_path, 'uncrop')

    res_331_crop, xception_331_crop, dense_331_crop, inception_331_crop, inceptionresnet_331_crop, efficient_331_crop = features.getAllModelFast(
        331, weight_path, 'crop')

    res_331_uncrop, xception_331_uncrop, dense_331_uncrop, inception_331_uncrop, inceptionresnet_331_uncrop, efficient_331_uncrop = features.getAllModelFast(
        331, weight_path, 'uncrop')

    model_list = [dense_224_uncrop,
                  dense_224_crop,
                  dense_331_uncrop,
                  dense_331_crop,
                  res_224_uncrop,
                  res_224_crop,
                  res_331_uncrop,
                  res_331_crop,
                  inception_224_uncrop,
                  inception_224_crop,
                  inception_331_uncrop,
                  inception_331_crop,
                  inceptionresnet_224_uncrop,
                  inceptionresnet_224_crop,
                  inceptionresnet_331_uncrop,
                  inceptionresnet_331_crop,
                  xception_224_uncrop,
                  xception_224_crop,
                  xception_331_uncrop,
                  xception_331_crop,
                  efficient_224_uncrop,
                  efficient_224_crop,
                  efficient_331_uncrop,
                  efficient_331_crop]

    return model_list


def get_pred(data_generators, model_list, weight_list, tta=False):
    """
    This function generates prediction for an image.
    
    Parameters:
        data_generators (list): a list of image data generators corresponding to 
        each model in model_list.
        
        model_list (list): a list of keras models used in ensembling.
        
        weight_list (list): a list of ensemble weight for each individual model.
        
        tta (boolean): a flag to check if test-time augmentation should be applied
        or not.
    
    Returns:
        ensemble_pred (float): the weighted sum of predicted probability generated
        from the ensembled model.
        
        ensemble_pred_round (int): the rounded weighted sum of predicted probability
        generated from the ensembled model. Can either be 1 or 0.
        
        predictions (list): a list of predicted probability of each ensembled model 
        for a single image.
    """
    
    combined_weighted_probs = []

    if tta:
        tta_steps = 10
        for model, datagen, weight in zip(model_list, data_generators, weight_list):
            predictions = []
            for i in tqdm(range(tta_steps)):
                preds = model.predict_generator(datagen, verbose=1)
                predictions.append(preds)
            Y_pred = np.mean(predictions, axis=0)
            y_pred = np.multiply(Y_pred, weight)
            combined_weighted_probs.append(y_pred)
    else:
        for model, datagen, weight in zip(model_list, data_generators, weight_list):
            Y_pred = model.predict_generator(datagen, verbose=1)
            y_pred = np.multiply(Y_pred, weight)
            combined_weighted_probs.append(y_pred)

    combined_weighted_probs = np.asarray(combined_weighted_probs)
    ensemble_pred = np.sum(combined_weighted_probs, axis=0)
    ensemble_pred_round = np.round(ensemble_pred)
    predictions = ['COVID-19 Positive' if pred == 1 else 'COVID-19 Negative' for pred in ensemble_pred_round]
    return ensemble_pred, ensemble_pred_round, predictions


def get_datagenerators_folder(image_path_224,
                              image_path_224_crop,
                              image_path_331,
                              image_path_331_crop,
                              tta=False):
    
    """
    This function creates a list of image data generators corresponds to each model
    used to ensemble using the path to directories with different types of augmentation
    (224x224 or 331x331 in size, cropped or uncopped).
    
    Parameters:
        image_path_224/331_crop/NONE (string): the path to images with different
        augmentations.
        
        tta (boolean): a flag to check if test-time augmentation should be applied
        or not.
    
    Returns:
        datagenlist (list): a list of image data generators corresponds to each model 
        used to ensemble.
    """

    img_util_224 = imgUtils(224)
    test_datagen_224_tta, test_datagen_224_notta = img_util_224.dataGen(rotation=15, h_shift=0.05, w_shift=0.05)

    img_util_331 = imgUtils(331)
    test_datagen_331_tta, test_datagen_331_notta = img_util_331.dataGen(rotation=15, h_shift=0.05, w_shift=0.05)

    if tta:
        test_datagen_224 = test_datagen_224_tta
        test_datagen_331 = test_datagen_331_tta
    else:
        test_datagen_224 = test_datagen_224_notta
        test_datagen_331 = test_datagen_331_notta

    images_224 = glob.glob(os.path.join(image_path_224, '*'))
    images_224 = pd.DataFrame.from_dict({'filename': images_224})

    images_224_crop = glob.glob(os.path.join(image_path_224_crop, '*'))
    images_224_crop = pd.DataFrame.from_dict({'filename': images_224_crop})

    images_331 = glob.glob(os.path.join(image_path_331, '*'))
    images_331 = pd.DataFrame.from_dict({'filename': images_331})

    images_331_crop = glob.glob(os.path.join(image_path_331_crop, '*'))
    images_331_crop = pd.DataFrame.from_dict({'filename': images_331_crop})

    batch_size = 100

    test_generator_224 = img_util_224.testgenerator_from_dataframe(batch_size=batch_size,
                                                                   test=test_datagen_224,
                                                                   dataframe=images_224,
                                                                   class_mode=None)

    test_generator_224_crop = img_util_224.testgenerator_from_dataframe(batch_size=batch_size,
                                                                        test=test_datagen_224,
                                                                        dataframe=images_224_crop,
                                                                        class_mode=None)

    test_generator_331 = img_util_331.testgenerator_from_dataframe(batch_size=batch_size,
                                                                   test=test_datagen_331,
                                                                   dataframe=images_331,
                                                                   class_mode=None)

    test_generator_331_crop = img_util_331.testgenerator_from_dataframe(batch_size=batch_size,
                                                                        test=test_datagen_331,
                                                                        dataframe=images_331_crop,
                                                                        class_mode=None)
    datagenlist = [test_generator_224,
                   test_generator_224_crop,
                   test_generator_331,
                   test_generator_331_crop]

    datagenlist = datagenlist * 6

    return datagenlist


def get_datagenerators_file(image_path_uncrop, image_path_crop, tta=False):
    """
    This function creates a list of image data generators corresponds to each model
    used to ensemble with different types of augmentation (cropped or uncopped).
    
    Parameters:
        image_path_crop/uncrop (string): the path to images with different
        augmentations.
        
        tta (boolean): a flag to check if test-time augmentation should be applied
        or not.
    
    Returns:
        datagenlist (list): a list of image data generators corresponds to each model 
        used to ensemble.
    """
    
    img_util_224 = imgUtils(224)
    img_util_331 = imgUtils(331)
    img_224 = img_util_224.proc_img(image_path_uncrop)
    img_331 = img_util_331.proc_img(image_path_uncrop)
    img_224_crop = img_util_224.proc_img(image_path_crop)
    img_331_crop = img_util_331.proc_img(image_path_crop)

    test_datagen_tta, test_datagen_no_tta = img_util_224.dataGen(rotation=15, h_shift=0.05, w_shift=0.05)

    if tta:
        test_datagen = test_datagen_tta
    else:
        test_datagen = test_datagen_no_tta

    test_generator_224 = test_datagen.flow(img_224,
                                           batch_size=1,
                                           shuffle=False)

    test_generator_224_crop = test_datagen.flow(img_224_crop,
                                                batch_size=1,
                                                shuffle=False)

    test_generator_331 = test_datagen.flow(img_331,
                                           batch_size=1,
                                           shuffle=False)

    test_generator_331_crop = test_datagen.flow(img_331_crop,
                                                batch_size=1,
                                                shuffle=False)
    datagenlist = [test_generator_224,
                   test_generator_224_crop,
                   test_generator_331,
                   test_generator_331_crop]

    datagenlist = datagenlist * 6

    return datagenlist


if __name__ == '__main__':
    """
    The main function preprocesses a test image dataset and generates predictions
    for the dataset. The results are saved as a .csv file. The images can optionally 
    be resized to 224x224 and 331x331. 

    """
    
    args = get_args()
    weights = os.path.normpath(args.weight_path)
    img_dir = os.path.normpath(args.img_path)
    pickle_path = os.path.normpath(args.ensemble_weights)
    ensemble_weights = pickle.load(open(pickle_path, "rb"))
    output_path = args.output
    if output_path is not None:
        output_path = os.path.normpath(output_path)
    tta = args.tta

    # Load models
    print('Loading models...')
    models = get_model_list(weights)
    print('Models loaded')

    # Analyze image/images
    runs = 1
    cont = 'yes'
    while cont == 'yes' or cont == 'y':
        if not os.path.isdir(img_dir):
            filename, ext = os.path.splitext(os.path.basename(img_dir))
            cropped_dir = get_crop(img_dir)
            img_cropped = os.path.join(cropped_dir, 'crop_squared', filename + '_crop_squared' + ext)
            img_uncropped = img_dir
            datagens = get_datagenerators_file(img_uncropped, img_cropped, tta=tta)
            files = [img_dir]
            ensemble_pred, binary_pred, predictions = get_pred(datagens, models, ensemble_weights, tta=tta)
            scores = ensemble_pred.flatten()
            binary_pred = binary_pred.flatten()
        else:
            uncropped_dir = img_dir
            cropped_dir = get_crop(img_dir)
            cropped_dir = os.path.join(cropped_dir, 'crop_squared')
            dir_224, dir_224_crop, dir_331, dir_331_crop = mkdirs(img_dir)
            resize_images(uncropped_dir, dir_224, 224)
            resize_images(cropped_dir, dir_224_crop, 224)
            resize_images(uncropped_dir, dir_331, 331)
            resize_images(cropped_dir, dir_331_crop, 331)
            datagens = get_datagenerators_folder(dir_224, dir_224_crop, dir_331, dir_331_crop, tta=tta)
            files = datagens[0].filenames
            ensemble_pred, binary_pred, predictions = get_pred(datagens, models, ensemble_weights, tta=tta)
            scores = ensemble_pred.flatten()
            binary_pred = binary_pred.flatten()

        # Print results and save to file (if output_path provided)
        for file, score, prediction in zip(files, scores, predictions):
            print(f'Prediction for {file} is {prediction}, with a raw score of {score}')
        if output_path is not None:
            results = pd.DataFrame.from_dict(
                {'Filenames': files, 'Scores': scores, 'Binary pred': binary_pred, 'Predictions': predictions})
            results.to_csv(os.path.join(output_path, 'predictions_' + str(runs) + '.csv'))
        runs += 1
        print('Predictions complete')

        # Prompt user to continue
        cont = input('Would you like to analyze another image/folder? (yes/no) ')
        if cont == 'yes' or cont == 'y':
            img_dir = input('Path to new image or folder of images? ')
        tf.keras.backend.clear_session()
        gc.collect()
    print('Done')

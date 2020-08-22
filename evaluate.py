import argparse
from utils import imgUtils
import pickle
from predict import get_model_list, get_pred
import os
import pandas as pd

def get_args():
    # Implement command line argument
    parser = argparse.ArgumentParser(
        description='For each input image, generates predictions of COVID-19 status.')
    parser.add_argument('--weight', '-w', dest='weight_path', metavar='weight_path',
                        type=str,
                        required=True, help='the path that contains trained weights.')

    parser.add_argument('--image', '-i', dest='img_path',
                        metavar='IMAGE_path', type=str,
                        required=True, help='the path to the image directory with subdirectory tree as indicated in '
                                            'the README file')

    parser.add_argument('-o', '--output', dest='output', metavar='prediction_output_path', type=str, required=True,
                        help='the directory to save results, including a csv of predictions, confusion matrix, '
                             'and ROC curve')

    parser.add_argument('--ensemble_weight', '-e', dest='ensemble_weights',
                        metavar='ensemble_weight_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'ensemble_weights.pickle'),
                        required=False, help='the path to the ensemble weights as a pickled list, '
                                             'if not supplied uses our pretrained weights by default')

    parser.add_argument('--tta', '-t', action='store_true', dest='tta',
                        help='switch to turn on test-time augmentation, warning: this takes significantly longer as '
                             'each model prediction is run 10 times')

    return parser.parse_args()


def locate_test_subdirs(img_dir):
    '''Identify the test dataset directories from a parent directory'''
    dir_224_uncrop = os.path.join(img_dir, '224', 'uncrop', 'Test')
    dir_224_crop = os.path.join(img_dir, '224', 'crop', 'Test')
    dir_331_uncrop = os.path.join(img_dir, '331', 'uncrop', 'Test')
    dir_331_crop = os.path.join(img_dir, '331', 'crop', 'Test')
    if not (os.path.exists(dir_224_uncrop) and os.path.exists(dir_224_crop)
            and os.path.exists(dir_331_uncrop) and os.path.exists(dir_331_uncrop)):
        print('Data path is invalid. Please check that directory tree is set up as described in README file.')
        exit()
    return dir_224_uncrop, dir_224_crop, dir_331_uncrop, dir_331_crop


def get_datagenerators_folders(image_path_224,
                              image_path_224_crop,
                              image_path_331,
                              image_path_331_crop,
                              tta=False,
                               batch_size = 100):
    '''
    Produces a list of datagenerators from paths to all 4 preprocessed versions of images
    Parameters:
        image_path_*: path to corresponding image data set
        tta: switch to turn on test time augmentation, default is off
        batch_size: mini-batch size for making predictions
    '''

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


    test_generator_224 = img_util_224.testgenerator_from_folder(batch_size=batch_size,
                                                                test=test_datagen_224,
                                                                data_dir=image_path_224)

    test_generator_224_crop = img_util_224.testgenerator_from_folder(batch_size=batch_size,
                                                                     test=test_datagen_224,
                                                                     data_dir=image_path_224_crop)

    test_generator_331 = img_util_331.testgenerator_from_folder(batch_size=batch_size,
                                                                test=test_datagen_331,
                                                                data_dir=image_path_331)

    test_generator_331_crop = img_util_331.testgenerator_from_folder(batch_size=batch_size,
                                                                     test=test_datagen_331,
                                                                     data_dir=image_path_331_crop)

    datagenlist = [test_generator_224,
                   test_generator_224_crop,
                   test_generator_331,
                   test_generator_331_crop]

    datagenlist = datagenlist * 6

    return datagenlist


if __name__ == '__main__':
    args = get_args()
    weights = os.path.normpath(args.weight_path)
    img_dir = os.path.normpath(args.img_path)
    pickle_path = os.path.normpath(args.ensemble_weights)
    ensemble_weights = pickle.load(open(pickle_path, "rb"))
    output_path = os.path.normpath(args.output)
    tta = args.tta

    #Load models
    print('Loading models...')
    models = get_model_list(weights)
    print('Models loaded')

    #Generate predictions
    dir_224_uncrop, dir_224_crop, dir_331_uncrop, dir_331_crop = locate_test_subdirs(img_dir)
    datagens = get_datagenerators_folders(dir_224_uncrop, dir_224_crop, dir_331_uncrop, dir_331_crop)
    scores, binary_pred, predictions = get_pred(datagens, models, ensemble_weights, tta=tta)
    scores = scores.flatten()
    binary_pred = binary_pred.flatten()
    actual = datagens[0].classes
    files = datagens[0].filenames
    for file, score, prediction in zip(files, scores, predictions):
        print(f'Prediction for {file} is  {prediction}, with a raw score of {score}.')

    #Save results
    results = pd.DataFrame.from_dict({'Filenames': files,
                                      'Scores': scores,
                                      'Binary prediction': binary_pred,
                                      'Predictions': predictions,
                                      'Actual': actual})
    results_path = os.path.join(output_path, 'Results')
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    results.to_csv(os.path.join(results_path, 'predictions.csv'))
    imgUtils.confusionMatrix(actual, binary_pred, results_path)
    imgUtils.ROCcurve(actual, scores, results_path)
    print('Done')



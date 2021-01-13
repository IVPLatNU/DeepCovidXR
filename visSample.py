# Examples of GradCAM visualization
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from vis.visualization import overlay
from utils import imgUtils, trainFeatures
import gc

"""Grad-CAM Visualization

This script will generate a grad-CAM image overlayed on the original image.

"""

def get_args():
    
    """
    This function gets various user input form command line. The user input variables
    include the path to the trained weight file, the path to a image and the path 
    where the result will be saved.
    
    Returns:
        parser.parse_args() (list): a list of user input values.
    """
    
    # Implement command line argument
    parser = argparse.ArgumentParser(description='For each input image, generates and saves a grad-CAM image.')
    parser.add_argument('--weight', '-w', dest='weight_path', metavar='weight_path',
                        type=str, nargs=1,
                        required=True, help='the path that contains trained weights.')
    parser.add_argument('--path', '-p', dest='img_path', metavar='img_path',
                        type=str, nargs=1, required=True, help='path to image for Grad-CAM')

    parser.add_argument('--output', '-o', dest='output', metavar='prediction_output_path', type=str,
                        default=None, help='the directory to save Grad-CAM results; if not provided will save to '
                                           'current working directory')
    return parser.parse_args()


def generate_gradCAM(input_path, img_size, models, output_path=os.getcwd()):
    
    """
    This function generates a Grad-CAM image for a given image. The Grad-CAM image
    is overlayed on the original image. The image will be demonstrated and saved
    in a given path.
    
    Parameters:
        input_path (string): the path to the image.
        img_size (int): the size of the input image.
        models (list): a list of models used to generate the Grad-CAM image.
        output_path (string): the path to save the output image.
        
    """
    
    img_list = []
    img_array_list = []
    img_name_list = []
    # current_path, input_filename = os.path.split(input_path)
    result_base = os.path.join(output_path, 'gradCAM_img')
    img_proc = imgUtils(img_size)

    # Create new folder to save the resulting image
    if not os.path.exists(result_base):
        os.mkdir(result_base)

    # Check if input path is a directory or file
    if os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            img_path = os.path.join(input_path, filename)
            input_img = image.load_img(img_path, target_size=(img_size, img_size),
                                       color_mode='rgb', interpolation='lanczos')
            img_array = np.asarray(input_img, dtype='float64')
            img_array_list.append(img_array)
            img_preproc = img_proc.preprocess(img_array)
            img = np.expand_dims(img_preproc, axis=0)
            img_list.append(img)

            # Split filename for result image name
            base_name = os.path.basename(filename)
            img_name = (os.path.splitext(base_name))[0]
            result_name = img_name + '_gradCAM.jpg'
            result_path = os.path.join(result_base, result_name)
            img_name_list.append(result_path)
    else:
        input_img = image.load_img(input_path, target_size=(img_size, img_size),
                                   color_mode='rgb', interpolation='lanczos')
        img_array = np.asarray(input_img, dtype='float64')
        img_array_list.append(img_array)
        img_preproc = img_proc.preprocess(img_array)
        img = np.expand_dims(img_preproc, axis=0)
        img_list.append(img)

        # Split filename for result image name
        base_name = os.path.basename(input_path)
        img_name, ext = os.path.splitext(base_name)
        result_name = img_name + '_gradCAM.png'
        result_path = os.path.join(result_base, result_name)
        img_name_list.append(result_path)

    i = 0
    for img in img_list:
        result_path = img_name_list[i]
        img_array = img_array_list[i]
        visualization = img_proc.gradCAM(img, models)

        plt.rcParams['figure.figsize'] = (18, 6)

        # Matplotlib preparations
        fig, axes = plt.subplots(1, 3)

        axes[0].imshow(img_array[..., 0], cmap='gray')
        axes[0].set_title('Input')
        axes[1].imshow(visualization)
        axes[1].set_title('Grad-CAM')
        heatmap = np.uint8(cm.jet(visualization)[..., :3] * 255)
        original = np.uint8(cm.gray(img_array[..., 0])[..., :3] * 255)
        axes[2].imshow(overlay(heatmap, original))
        axes[2].set_title('Overlay')
        plt.savefig(result_path)
        i += 1


if __name__ == '__main__':
    
    """
    The main function load all models and generates an overlayed Grad-CAM image.
    This process can be repeated on multiple images.
    
    """
    
    args = get_args()
    input_path = os.path.normpath(args.img_path[0])
    weights = os.path.normpath(args.weight_path[0])
    output_path = args.output
    if output_path is not None:
        output_path = os.path.normpath(output_path)
    else:
        output_path = os.getcwd()

    crop_stat = 'uncrop'
    img_size = 224

    print('Loading models...')
    features = trainFeatures()
    res_224, xception_224, dense_224, inception_224, inceptionres_224, efficient_224 = features.getAllModel(img_size,
                                                                                                            weights,
                                                                                                            crop_stat)
    models = [res_224, xception_224, dense_224, inception_224, inceptionres_224, efficient_224]
    print('Models loaded')

    cont = 'yes'
    while cont == 'yes' or cont == 'y':
        print('Computing grad-CAM heatmap...')
        generate_gradCAM(input_path, img_size, models, output_path=output_path)
        print('Heatmap generated')
        cont = input('Would you like to analyze another image/folder? (yes/no) ')
        if cont == 'yes' or cont == 'y':
            input_path = input('Path to new image or folder of images? ')
        gc.collect()
    print('Done')

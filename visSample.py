# Examples of GradCAM visualization
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from vis.visualization import  overlay
from utils import imgUtils, trainFeatures

def get_args():
    # Implement command line argument
    parser = argparse.ArgumentParser(description='For each input image, generates and saves a grad-CAM image.')
    parser.add_argument('--weight', '-w', dest = 'weight_path', metavar = 'weight_path', 
                        type = str, nargs = 1,
                        required = True, help='the path that contains trained weights.')
    parser.add_argument('--path', '-p', dest = 'img_path', metavar = 'img_path', 
                        type = str, nargs = 1)
    
    return parser.parse_args()

if __name__=='__main__':
    args = get_args()
    weights = args.weight_path[0]
    input_path = args.img_path[0]

#     weights = r'D:\covid\Ensemble\covid_weights'
#     input_path = r'C:\Users\sheng\OneDrive\文档\GitHub\deepcovidxr\sample_images'

    img_size = 224
    crop_stat = 'uncrop'
    img_list = []
    img_preproc_list = []
    img_name_list = []
    current_path = os.getcwd()
    result_base = os.path.join(current_path, 'gradCAM_img')
    
    features = trainFeatures()
    img_proc = imgUtils(img_size)
    
    # Create new folder to save the resulting image
    if not os.path.exists(result_base):
        os.mkdir(result_base)
        
    # Check if input path is a directory or file
    if os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            img_path = os.path.join(input_path, filename)
            input_img = image.load_img(img_path, target_size = (img_size, img_size), 
                                    color_mode='rgb', interpolation = 'lanczos')
            img_array = np.asarray(input_img, dtype = 'float64')
            img_preproc = img_proc.preprocess(img_array)
            img_preproc_list.append(img_preproc)
            img = np.expand_dims(img_preproc, axis = 0)
            img_list.append(img)
            
            # Split filename for result image name
            base_name = os.path.basename(filename)
            img_name = (os.path.splitext(base_name))[0]
            result_name = img_name + '_gradCAM.jpg'
            result_path = os.path.join(result_base, result_name)
            img_name_list.append(result_path)
    else:
        input_img = image.load_img(img_path, target_size = (img_size, img_size), 
                                    color_mode='rgb', interpolation = 'lanczos')
        img_array = np.asarray(input_img, dtype = 'float64')
        img_preproc = img_proc.preprocess(img_array)
        img_preproc_list.append(img_preproc)
        img = np.expand_dims(img_preproc, axis = 0)
        img_list.append(img)
        
        # Split filename for result image name
        base_name = os.path.basename(input_path)
        img_name = os.path.splitext(base_name)
        result_name = img_name + '_gradCAM.jpg'
        result_path = os.path.join(result_base, result_name)
        img_name_list.append(result_name)

    print('Loading models...')
    res_224, xception_224, dense_224, inception_224, inceptionres_224, efficient_224 = features.getAllModel(img_size, weights, crop_stat)
    models = [res_224, xception_224, dense_224, inception_224, inceptionres_224, efficient_224]
    print('Models loaded')

    i = 0
    for img in img_list:
        img_preproc = img_preproc_list[i]
        result_path = img_name_list[i]

        visualization_avg, heatmap = img_proc.gradCAM(img, img_preproc, models)
        i = i + 1
    
        plt.rcParams['figure.figsize'] = (18, 6)
    
        # Matplotlib preparations
        fig, axes = plt.subplots(1, 3)
        
        axes[0].imshow(img_array[..., 0], cmap='gray') 
        axes[0].set_title('Input')
        
        original = np.uint8(cm.gray(img_array[..., 0])[..., :3]*255)
        axes[1].imshow(heatmap)
        axes[1].set_title('Grad-CAM')
        axes[2].imshow(overlay(heatmap, original))
        axes[2].set_title('Overlay')
        plt.savefig(result_path) 

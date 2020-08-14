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
    parser.add_argument('--weight', '-w', dest='weight_path', metavar='weight_path', 
                        type=str, nargs=1,
                        required = True, help='the path that contains trained weights.')
    
    return parser.parse_args()

if __name__=='__main__':
    args = get_args()
    weights = args.weight_path[0]
    #weights = 'D:\covid\Ensemble\covid_weights'
    img_size = 224
    dropout = 0.3
    crop_stat = 'uncrop'
        
    features = trainFeatures()
    img_proc = imgUtils(img_size)
    
    current_path = os.getcwd()
    img_name = 'pos_sample1.jpg'
    img_path = os.path.join(current_path, 'sample_images', img_name)
    original_image = image.load_img(img_path, target_size = (img_size, img_size), 
                                    color_mode='rgb', interpolation = 'lanczos')
    img_array = np.asarray(original_image, dtype='float64')
    img_preproc = img_proc.preprocess(img_array)
    img = np.expand_dims(img_preproc, axis=0)

    res_224, xception_224, dense_224, inception_224, inceptionres_224, efficient_224 = features.getAllModel(img_size, weights, crop_stat)
    models = [res_224, xception_224, dense_224, inception_224, inceptionres_224, efficient_224]
    result_name = 'sample1_gradcam.jpg'
    result_path = os.path.join(os.getcwd(),'sample_images', result_name)
    # Save overlay heatmap
    visualization_avg = img_proc.gradCAM(img, img_preproc, models)
    
    plt.rcParams['figure.figsize'] = (18, 6)

    # Matplotlib preparations
    fig, axes = plt.subplots(1, 3)
    
    axes[0].imshow(img_array[..., 0], cmap='gray') 
    axes[0].set_title('Input')
    
    heatmap_avg = np.uint8(visualization_avg)
    original = np.uint8(cm.gray(img_array[..., 0])[..., :3]*255)
    axes[1].imshow(heatmap_avg)
    axes[1].set_title('Grad-CAM')
    axes[2].imshow(overlay(heatmap_avg, original))
    axes[2].set_title('Overlay')
    plt.show()




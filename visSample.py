# Examples of GradCAM visualization
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import argparse

from utils import imgUtils, trainFeatures

def get_args():
    # Implement command line argument
    parser = argparse.ArgumentParser(description='For each input image, generates COVID possibilities for 224x224 and 331x331 versions.')
    parser.add_argument('--weight', '-w', dest='weight_path', metavar='weight_path', 
                        type=str, nargs=1,
                        required = True, help='the path that contains trained weights.')
    
    return parser.parse_args()

if __name__=='__main__':
    args = get_args()
    weights = args.weight_path[0]
#    weights = 'D:\covid\Ensemble\covid_weights'
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

    res_224, xception_224, dense_224, inception_224, inceptionres_224, efficient_224 = features.getAllDropModel(img_size, weights, crop_stat, dropout)
    models = [res_224, xception_224, dense_224, inception_224, inceptionres_224, efficient_224]
    
    # Save overlay heatmap
    visualization = img_proc.gradCAM(img, img_preproc, models)




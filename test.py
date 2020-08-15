# Test individual image

import argparse
from utils import imgUtils, trainFeatures
import numpy as np
import pickle
#import time
from tqdm import tqdm
import os
import cv2
from PIL import Image

img_size1 = 224
img_size2 = 331
# =============================================================================
# 
# def get_args():
#     # Implement command line argument
#     parser = argparse.ArgumentParser(description='For each input image, generates COVID possibilities for 224x224 and 331x331 versions.')
#     parser.add_argument('--weight', '-w', dest='weight_path', metavar='weight_path', 
#                         type=str, nargs=1,
#                         required = True, help='the path that contains trained weights.')
#     
#     parser.add_argument('--image', '-i', dest='img_path', 
#                         metavar='IMAGE_path', type=str, nargs=1,
#                         required = True, help='the path to the image.')
#     
#     return parser.parse_args()
# =============================================================================

def mkdir_224(img_dir):
    dir_name = os.path.join(img_dir, '224')
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        
    return dir_name

def mkdir_331(img_dir):
    dir_name = os.path.join(img_dir, '331')
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        
    return dir_name

def get_crop(input_folder):
    # Run Crop_img to get a new cropped image
    current_path = os.getcwd()
    unet_path = os.path.join(current_path, 'trained_unet_model.hdf5')
    unet_path = r'D:\covid\trained_unet_model.hdf5'
    input_folder_224 = os.path.join(input_folder, '224')
    input_folder_331 = os.path.join(input_folder, '331')
    output_dir_224 = os.path.join(input_folder, 'cropped_test_img_224')
    output_dir_331 = os.path.join(input_folder, 'cropped_test_img_331')
    if not os.path.exists(output_dir_224):
        os.mkdir(output_dir_224)
    if not os.path.exists(output_dir_331):
        os.mkdir(output_dir_331)
    
    command_line_224 = 'python Crop_img.py -U ' + unet_path + ' -o ' + output_dir_224 + ' -f ' + input_folder_224
    command_line_331 = 'python Crop_img.py -U ' + unet_path + ' -o ' + output_dir_331 + ' -f ' + input_folder_331
    os.system(command_line_224)
    os.system(command_line_331)
    return output_dir_224, output_dir_331
    

def get_model_list(weight_path):
    model_list = []
    features = trainFeatures()
    res_224_crop, xception_224_crop, dense_224_crop, inception_224_crop, inceptionresnet_224_crop, efficient_224_crop = features.getAllModel(img_size1, weight_path, 'crop')
    
    res_224_uncrop, xception_224_uncrop, dense_224_uncrop, inception_224_uncrop, inceptionresnet_224_uncrop, efficient_224_uncrop = features.getAllModel(img_size1, weight_path, 'uncrop')
    
    res_331_crop, xception_331_crop, dense_331_crop, inception_331_crop, inceptionresnet_331_crop, efficient_331_crop = features.getAllModel(img_size2, weight_path, 'crop')
    
    res_331_uncrop, xception_331_uncrop, dense_331_uncrop, inception_331_uncrop, inceptionresnet_331_uncrop, efficient_331_uncrop = features.getAllModel(img_size2, weight_path, 'uncrop')

    model_list.extend([dense_224_uncrop, 
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
              efficient_331_crop])
        
    return model_list

def get_pred(input_img, model_list, weight_list):
    tta_steps = 5
    combined_weighted_probs = []
    combined_probs = []
    
    for model, weight in zip(model_list, weight_list):
        predictions = []
        for i in tqdm(range(tta_steps)):
            preds = model.predict(input_img, verbose=1)
            predictions.append(preds)
        Y_pred = np.mean(predictions, axis=0)
        y_pred = np.multiply(Y_pred, weight)
        combined_probs.append(Y_pred)
        combined_weighted_probs.append(y_pred)
        
    combined_probs = np.asarray(combined_probs)
    combined_weighted_probs = np.asarray(combined_weighted_probs)
    
    ensemble_pred = np.sum(combined_weighted_probs)
    return ensemble_pred

def test_individual(model_list, pickle_path, input_img_224, input_img_224_crop, 
                    input_img_331, input_img_331_crop):
    
    ensemble_weights = pickle.load(open(pickle_path, "rb"))

    model_list_224_uncrop = model_list[0]+model_list[4]+model_list[8]+model_list[12]+model_list[16]+model_list[20]
    model_list_224_crop = model_list[1]+model_list[5]+model_list[9]+model_list[13]+model_list[17]+model_list[21]
    model_list_331_uncrop = model_list[2]+model_list[6]+model_list[10]+model_list[14]+model_list[18]+model_list[22]
    model_list_331_crop = model_list[3]+model_list[7]+model_list[11]+model_list[15]+model_list[19]+model_list[23]
    ensemble_weights_224_uncrop = ensemble_weights[0]+ensemble_weights[4]+ensemble_weights[8]+ensemble_weights[12]+ensemble_weights[16]+ensemble_weights[20]
    ensemble_weights_224_crop = ensemble_weights[1]+ensemble_weights[5]+ensemble_weights[9]+ensemble_weights[13]+ensemble_weights[17]+ensemble_weights[21]
    ensemble_weights_331_uncrop = ensemble_weights[2]+ensemble_weights[6]+ensemble_weights[10]+ensemble_weights[14]+ensemble_weights[18]+ensemble_weights[22]
    ensemble_weights_331_crop = ensemble_weights[3]+ensemble_weights[7]+ensemble_weights[11]+ensemble_weights[15]+ensemble_weights[19]+ensemble_weights[23]
    
    pred_224_uncrop = get_pred(input_img_224, model_list_224_uncrop, ensemble_weights_224_uncrop)
    pred_331_uncrop = get_pred(input_img_331, model_list_331_uncrop, ensemble_weights_331_uncrop)
    pred_224_crop = get_pred(input_img_224_crop, model_list_224_crop, ensemble_weights_224_crop)
    pred_331_crop = get_pred(input_img_331_crop, model_list_331_crop, ensemble_weights_331_crop)
    
    ensemble_pred = np.mean(pred_224_uncrop, pred_224_crop, pred_331_uncrop, pred_331_crop)
    
    return ensemble_pred

if __name__=='__main__':
# =============================================================================
#     args = get_args()
#     weights = args.weight_path[0]
#     img_dir = args.img_path[0]
# =============================================================================
    weights = r'D:\covid\Ensemble\covid_weights'
    img_dir = r'D:\covid\sample_images\pos_sample1.png'
    if not os.path.isdir(img_dir):
        img_dir1 = os.path.dirname(img_dir)
    pickle_path = 'ensemble_weights.pickle'
    
    img_list_224 = []
    img_list_331 = []
    img_list_224_crop = []
    img_list_331_crop = []
    img_name_list = []
    
    img_util_224 = imgUtils(224)
    img_util_331 = imgUtils(331)
    
    # Prepare for cropping
    
    if os.path.isdir(img_dir):
        dir_224 = mkdir_224(img_dir)
        dir_331 = mkdir_331(img_dir)
        for filename in os.listdir(img_dir):
            base_name = os.path.basename(filename)
            img_name_list.append(base_name)
            img_name = (os.path.splitext(base_name))[0]
            img_name_224 = img_name + '_224'+(os.path.splitext(base_name))[1]
            img_name_331 = img_name + '_331'+(os.path.splitext(base_name))[1]
            img_224 = img_util_224.proc_img(filename)
            img_331 = img_util_331.proc_img(filename)
            
            out_224_name = os.path.join(dir_224, img_name_224)
            out_331_name = os.path.join(dir_331, img_name_331)
            cv2.imwrite(out_224_name, img_224)
            cv2.imwrite(out_331_name, img_331)
            
            output_dir_224, output_dir_331 = get_crop(img_dir)
            
    else:
        dir_224 = mkdir_224(img_dir1)
        dir_331 = mkdir_331(img_dir1)
        base_name = os.path.basename(img_dir)
        img_name_list.append(base_name)
        img_name = (os.path.splitext(base_name))[0]
        img_name_224 = img_name + '_224'+(os.path.splitext(base_name))[1]
        img_name_331 = img_name + '_331'+(os.path.splitext(base_name))[1]
# =============================================================================
#         img_224 = (img_util_224.proc_img(img_dir))[0,...]
#         img_331 = (img_util_331.proc_img(img_dir))[0,...]
# =============================================================================
        im = Image.open(img_dir)
        im_224 = im.resize((224,224), resample=Image.LANCZOS)
        im_331 = im.resize((331, 331), resample = Image.LANCZOS)       
        out_224_name = os.path.join(dir_224, img_name_224)
        out_331_name = os.path.join(dir_331, img_name_331)
        im_224.save(out_224_name)
        im_331.save(out_331_name)
# =============================================================================
#         cv2.imwrite(out_224_name, img_224)
#         cv2.imwrite(out_331_name, img_331)
# =============================================================================
            
        # Perform cropping on 224 and 331 images
        output_dir_224, output_dir_331 = get_crop(img_dir1)
    
    models = get_model_list(weights)
    output_dir_224_squared = os.path.join(output_dir_224, 'crop_squared')
    output_dir_331_squared = os.path.join(output_dir_331, 'crop_squared')
    
    for filename in os.listdir(dir_224):
        img_224 = img_util_224.proc_img(os.path.join(dir_224, filename))
        img_list_224.append(img_224)
    
    for filename in os.listdir(output_dir_224_squared):
        img_224 = img_util_224.proc_img(os.path.join(output_dir_224_squared, filename))
        img_list_224_crop.append(img_224)
        
    for filename in os.listdir(dir_331):
        img_331 = img_util_331.proc_img(os.path.join(dir_331, filename))
        img_list_331.append(img_331)
        
    for filename in os.listdir(output_dir_331_squared):
        img_331 = img_util_331.proc_img(os.path.join(output_dir_331_squared, filename))
        img_list_331_crop.append(img_331)
        
    i = 0
        
    for img_224 in img_list_224:
        img_name = img_name_list[i]
        img_331 = img_list_331[i]
        img_224_crop = img_list_224_crop[i]
        img_331_crop = img_list_331_crop[i]
        score = test_individual(models, pickle_path, img_224, img_224_crop, 
                                           img_331, img_331_crop) 
        print('Prediction for'+ img_name + ' is {pred1:.3f}.'.format(pred1 = score))
        i = i+1
#    end_time = time.time()
    
#    print('Prediction for 224x224 image is {pred1:.3f}.\nPrediction for 331x331 image is {pred2:.3f}\n'.format(pred1 = score_224, pred2 = score_331))
#    print('Time used is {time_used}.'.format(time_used = end_time - start_time))




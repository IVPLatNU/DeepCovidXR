# Test individual image

import argparse
from utils import imgUtils, trainFeatures
import numpy as np
import pickle
import time
from tqdm import tqdm

from Crop_img import single_img_crop

img_size1 = 224
img_size2 = 331

def get_args():
    # Implement command line argument
    parser = argparse.ArgumentParser(description='For each input image, generates COVID possibilities for 224x224 and 331x331 versions.')
    parser.add_argument('--weight', '-w', dest='weight_path', metavar='weight_path', 
                        type=str, nargs=1,
                        required = True, help='the path that contains trained weights.')
    
    parser.add_argument('--image', '-i', dest='img_path', 
                        metavar='IMAGE_path', type=str, nargs=1,
                        required = True, help='the path to the image.')
    
    return parser.parse_args()

# Resize image to 224 and 331
def img_resize(img, img_size):
    i = 0
    
    

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
# =============================================================================
#     tta_steps = 5
#     combined_weighted_probs = []
#     combined_probs = []
#     
#     for model, weight in zip(model_list_224_uncrop, ensemble_weights_224_uncrop):
#         predictions = []
#         for i in tqdm(range(tta_steps)):
#             preds = model.predict(input_img_224, verbose=1)
#             predictions.append(preds)
#         Y_pred = np.mean(predictions, axis=0)
#         y_pred = np.multiply(Y_pred, weight)
#         combined_probs.append(Y_pred)
#         combined_weighted_probs.append(y_pred)
#         
#     combined_probs = np.asarray(combined_probs)
#     combined_weighted_probs = np.asarray(combined_weighted_probs)
#     
#     ensemble_pred_224_uncrop = np.sum(combined_weighted_probs)
#     
#     combined_probs = []
#     combined_weighted_probs = []
#         
#     for model, weight in zip(model_list_331_uncrop, ensemble_weights_331_uncrop):
#         predictions = []
#         for i in tqdm(range(tta_steps)):
#             preds = model.predict(input_img_331, verbose=1)
#             predictions.append(preds)
#         Y_pred = np.mean(predictions, axis=0)
#         y_pred = np.multiply(Y_pred, weight)
#         combined_probs.append(Y_pred)
#         combined_weighted_probs.append(y_pred)
#     
#     combined_probs = np.asarray(combined_probs)
#     combined_weighted_probs = np.asarray(combined_weighted_probs)
#     
#     ensemble_pred_331_uncrop = np.sum(combined_weighted_probs)
# =============================================================================

if __name__=='__main__':
    args = get_args()
    weights = args.weight_path[0]
    img_dir = args.img_path[0]
    
    pickle_path = 'ensemble_weights.pickle'
    start_time = time.time()
    
    img_util_224 = imgUtils(224)
    img_util_331 = imgUtils(331)
    
    img_224 = img_util_224.proc_img(img_dir)
    img_331 = img_util_331.proc_img(img_dir)
    models = get_model_list(weights)
    score_224, score_331 = test_individual(models, pickle_path, img_224, img_331) 
    
    end_time = time.time()
    
    print('Prediction for 224x224 image is {pred1:.3f}.\nPrediction for 331x331 image is {pred2:.3f}\n'.format(pred1 = score_224, pred2 = score_331))
    print('Time used is {time_used}.'.format(time_used = end_time - start_time))




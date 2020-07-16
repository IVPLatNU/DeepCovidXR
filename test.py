# Test individual image

import argparse
import os
from utils import imgUtils, trainFeatures
import numpy as np
from deepstack.ensemble import StackEnsemble
from sklearn.ensemble import RandomForestRegressor
from deepstack.ensemble import DirichletEnsemble
from deepstack.base import KerasMember

def get_args():
    # Implement command line argument
    parser = argparse.ArgumentParser(description='Ensemble trained models to generate confusion matrices.')
    parser.add_argument('--weight', '-w', dest='weight_path', metavar='weight_path', 
                        type=str, nargs=1,
                        required = True, help='the path that contains trained weights.')
    
    parser.add_argument('--image', '-i', dest='img_path', 
                        metavar='IMAGE_path', type=str, nargs=1,
                        required = True, help='the path to the image.')
    
    parser.add_argument('--size', '-s', dest='img_size', 
                        metavar='IMAGE_SIZE', type=str, nargs=1,
                        required = True, help='the size of the image.')
    
    return parser.parse_args()

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
    

def get_members(ntta_generator_list):
    model_list = []
    model_name_list = []
    member_list = []
    features = trainFeatures()
    res_224_crop, xception_224_crop, dense_224_crop, inception_224_crop, inceptionres_224_crop, efficient_224_crop = features.getAllModel(img_size1, weights, 'crop')
    model_list.extend([res_224_crop, xception_224_crop, dense_224_crop, 
                       inception_224_crop, inceptionres_224_crop, efficient_224_crop])
    model_name_list.extend(["res_224_crop", "xception_224_crop", "dense_224_crop", 
                       "inception_224_crop", "inceptionres_224_crop", "efficient_224_crop"])
    
    res_224_uncrop, xception_224_uncrop, dense_224_uncrop, inception_224_uncrop, inceptionres_224_uncrop, efficient_224_uncrop = features.getAllModel(img_size1, weights, 'uncrop')
    model_list.extend([res_224_uncrop, xception_224_uncrop, dense_224_uncrop, 
                       inception_224_uncrop, inceptionres_224_uncrop, efficient_224_uncrop])
    model_name_list.extend(["res_224_uncrop", "xception_224_uncrop", "dense_224_uncrop", 
                       "inception_224_uncrop", "inceptionres_224_uncrop", "efficient_224_uncrop"])
    
    res_331_crop, xception_331_crop, dense_331_crop, inception_331_crop, inceptionres_331_crop, efficient_331_crop = features.getAllModel(img_size2, weights, 'crop')
    model_list.extend([res_331_crop, xception_331_crop, dense_331_crop, 
                       inception_331_crop, inceptionres_331_crop, efficient_331_crop])
    model_name_list.extend(["res_331_crop", "xception_331_crop", "dense_331_crop", 
                       "inception_331_crop", "inceptionres_331_crop", "efficient_331_crop"])
    
    res_331_uncrop, xception_331_uncrop, dense_331_uncrop, inception_331_uncrop, inceptionres_331_uncrop, efficient_331_uncrop = features.getAllModel(img_size2, weights, 'uncrop')
    model_list.extend([res_331_uncrop, xception_331_uncrop, dense_331_uncrop, 
                       inception_331_uncrop, inceptionres_331_uncrop, efficient_331_uncrop])
    model_name_list.extend(["res_331_uncrop", "xception_331_uncrop", "dense_331_uncrop", 
                       "inception_331_uncrop", "inceptionres_331_uncrop", "efficient_331_uncrop"])
    
    for i in range(24):
        member = create_member(model_name_list[i], model_list[i], ntta_generator_list)
        member_list.append(member)
        
    return member_list, model_list, model_name_list

def preprocess_img():
    



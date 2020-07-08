# Build EfficientNet model

from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import efficientnet.tfkeras as efn 


class EfficientNet():
    def _init_(self, weights):
        self.weights = weights
        
    def build_model(self, img_size):
        base_model = efn.EfficientNetB2(weights=self.weights, include_top=False, 
                                        backend = keras.backend, layers = keras.layers, 
                                        models = keras.models, utils = keras.utils,
                                        input_shape = (img_size,img_size,3))
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        predictions = layers.Dense(1, activation='sigmoid', name='last')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def freeze(self, model):
        for layer in model.layers[:332]:
            layer.trainable = False
        for layer in model.layers[332:]:
            layer.trainable = True
            
        return model
    
    def unfreeze(self, model):
        for layer in model.layers[0:]:
            layer.trainable = True
        return model


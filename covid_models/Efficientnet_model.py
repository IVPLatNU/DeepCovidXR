# Build EfficientNet model

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import efficientnet.tfkeras as efn
from tensorflow.keras.models import load_model

class EfficientNet():
    def __init__(self, weights):
        self.weights = weights
        
    def buildBaseModel(self, img_size):
        base_model = efn.EfficientNetB2(weights='imagenet', include_top=False, 
                                        backend = keras.backend, layers = keras.layers, 
                                        models = keras.models, utils = keras.utils,
                                        input_shape = (img_size,img_size,3))
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        predictions = layers.Dense(1, activation='sigmoid', name='last')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(self.weights)
        return model

    def buildBaseModelFast(self):
        model = load_model(self.weights, compile=False)
        return model
    
    def buildNihModel(self, img_size, label_len):
        base_model = efn.EfficientNetB2(weights='imagenet', include_top=False, 
                                 input_shape = (img_size,img_size,3))
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        predictions = layers.Dense(label_len, activation='sigmoid', name='last')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        if not self.weights == 'imagenet':
            model.load_weights(self.weights)
        return model
    
    def buildTunerModel(self, img_size):
        base_model = efn.EfficientNetB2(weights='imagenet', include_top=False, 
                                        backend = keras.backend, layers = keras.layers, 
                                        models = keras.models, utils = keras.utils,
                                        input_shape = (img_size,img_size,3))
        base_model.load_weights(self.weights, by_name = True)
        
        return base_model
    
    def freeze(self, model):
        for layer in model.layers[:332]:
            layer.trainable = False
        for layer in model.layers[332:]:
            layer.trainable = True
            
        return model
    
    def buildDropModel(self, img_size, dropout):
        base_model = efn.EfficientNetB2(weights=None, include_top=False, 
                                 input_shape = (img_size,img_size,3))
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout)(x)
        predictions = layers.Dense(1, activation='sigmoid', name='last')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(self.weights)
        return model


# Build Densenet model

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121

class DenseNet():
    def __init__(self, weights):
        self.weights = weights
        
    def buildBaseModel(self, img_size):
        base_model = DenseNet121(weights='imagenet', include_top=False, 
                                 input_shape = (img_size,img_size,3))
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        predictions = layers.Dense(1, activation='sigmoid', name='last')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(self.weights)
        return model
    
    def buildTunerModel(self, img_size):
        base_model = DenseNet121(weights=self.weights, include_top=False, 
                                 input_shape = (img_size,img_size,3))
        return base_model
    
    def freeze(self, model):
        for layer in model.layers[:428]:
            layer.trainable = False
        for layer in model.layers[428:]:
            layer.trainable = True
            
        return model
    
    def buildDropModel(self, img_size, dropout):
        base_model = DenseNet121(weights=self.weights, include_top=False, 
                                 input_shape = (img_size,img_size,3))
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout)(x)
        predictions = layers.Dense(1, activation='sigmoid', name='last')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    


# Build InceptionNet model

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import optimizers


class InceptionNet():
    def __init__(self, weights):
        self.weights = weights
        
    def buildBaseModel(self, img_size, dropout_rate):
        base_model = InceptionV3(weights=self.weights, include_top=False, 
                         input_shape = (img_size,img_size,3))
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        predictions = layers.Dense(1, activation='sigmoid', name='last')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def freeze(self, model):
        for layer in model.layers[:310]:
            layer.trainable = False
        for layer in model.layers[310:]:
            layer.trainable = True
            
        return model
    
    def unfreeze(self, model):
        for layer in model.layers[0:]:
            layer.trainable = True
        return model
    
    def compileModel(self, model, lr, momentum, nestrov):
        model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(learning_rate=lr, 
                                             momentum=momentum,
                                             nesterov=nestrov,
                                             ), 
              metrics=['acc', 
                    tf.keras.metrics.AUC(name='auc'), 
                    tf.keras.metrics.Precision(name='precision'), 
                    tf.keras.metrics.Recall(name='recall')])


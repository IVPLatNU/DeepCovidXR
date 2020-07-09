# Build hyper model for tuner 

import tensorflow as tf
from kerastuner import HyperModel
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers

class hyperModel(HyperModel):
    
    def __init__(self, model, weight):
        self.model = model
        self.weight = weight
        
    def build(self, hp):
        base_model = self.model
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(hp.Float('dropout_rate',
                                            min_value=0,
                                            max_value=0.5,
                                            default=0))(x)
        predictions = layers.Dense(1, activation='sigmoid', name='last')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(self.weight)

        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(
                                    learning_rate=hp.Float('learning_rate',
                                            min_value=0.001,
                                            max_value=0.01,
                                            default=0.5),
                                    momentum=hp.Float('momentum',
                                            min_value=0.5,
                                            max_value=0.9,
                                            default=0.5)),
            metrics=['acc', 
                       tf.keras.metrics.AUC(name='auc'), 
                       tf.keras.metrics.Precision(name='precision'), 
                       tf.keras.metrics.Recall(name='recall')])
        return model
        

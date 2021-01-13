# Build hyper model for tuner 

import tensorflow as tf
from kerastuner import HyperModel
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers

class hyperModel(HyperModel):
    """
    This is a class for building keras hyperparameter tuner model.
    
    """    
    def __init__(self, model, weight):
        """ 
        The constructor for hyperModel class. 
  
        Parameters: 
            model (class): the base model used to build the tuner model.
            weight (string): the path to a pretrained weight file.     
        """
        self.model = model
        self.weight = weight
        
    def build(self, hp):
        """
        This function builds a keras tuner model based on the model provided.
        The tuned hyper parameter includes dropout rate, learning rate and momentum.
        
        Parameters:
            hp (class):  where hyperparameters can be sampled from.

        Returns:
            model (class): the hyperparameter tuner model used in training.

        """
        base_model = self.model
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(hp.Float('dropout_rate',
                                            min_value=0,
                                            max_value=0.5,
                                            default=0.3))(x)
        predictions = layers.Dense(1, activation='sigmoid', name='last')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(self.weight)

        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(
                                    learning_rate=hp.Float('learning_rate',
                                            min_value=0.00001,
                                            max_value=0.1,
                                            default=0.01),
                                    momentum=hp.Float('momentum',
                                            min_value=0,
                                            max_value=0.9,
                                            default=0.5)),
            metrics=['acc', 
                       tf.keras.metrics.AUC(name='auc'), 
                       tf.keras.metrics.Precision(name='precision'), 
                       tf.keras.metrics.Recall(name='recall')])
        return model
        

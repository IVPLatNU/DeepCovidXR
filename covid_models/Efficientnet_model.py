# Build EfficientNet model

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import efficientnet.tfkeras as efn
from tensorflow.keras.models import load_model

class EfficientNet():
    """
    This is a class for building various EfficientNet-B2 model for different usage,
    including a pretraining model, a training model, a model for keras tuner and
    a model with dropout layer.
    
    """
    def __init__(self, weights):
        """ 
        The constructor for EfficientNet class. 
  
        Parameters: 
           weight (string): the path to a pretrained weight file.     
        """
        self.weights = weights
        
    def buildBaseModel(self, img_size):
        """
        This function builds a EfficientNet-B2 model which includes a global
        average pooling layer and a dense layer with sigmoid activation function.
        
        Parameters:
            img_size (int): the size of input images (img_size, img_size).

        Returns:
            model (class): the base Efficient-B2 model that can be used later in training.

        """
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
        """
        This function loads a EfficientNet-B2 model.

        Returns:
            model (class): the model with weights loaded.

        """
        model = load_model(self.weights, compile=False)
        return model
    
    def buildNihModel(self, img_size, label_len):
        """
        This function builds a base DenseNet-121 model for pretraining with the NIH
        dataset.
        
        Parameters:
            img_size (int): the size of input images (img_size, img_size).
            label_len (int): the length of the labels from the NIH dataset.

        Returns:
            model (class): the EfficientNet-B2 model used in pretraining.

        """
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
        """
        This function builds a base EfficientNet-B2 model for keras tuner
        
        Parameters:
            img_size (int): the size of input images (img_size, img_size).

        Returns:
            model (class): the EfficientNet-B2 model used for keras tuner.

        """
        base_model = efn.EfficientNetB2(weights='imagenet', include_top=False, 
                                        backend = keras.backend, layers = keras.layers, 
                                        models = keras.models, utils = keras.utils,
                                        input_shape = (img_size,img_size,3))
        base_model.load_weights(self.weights, by_name = True)
        
        return base_model
    
    def freeze(self, model):
        """
        This function builds a EfficientNet-B2 model with layers other than fully 
        connected layers freezed.
        
        Parameters:
            img_size (int): the size of input images (img_size, img_size).

        Returns:
            model (class): the EfficientNet-B2 model with fully connected layers as 
            only trainable layers.

        """
        for layer in model.layers[:332]:
            layer.trainable = False
        for layer in model.layers[332:]:
            layer.trainable = True
            
        return model
    
    def buildDropModel(self, img_size, dropout):
        """
        This function builds a EfficientNet-B2 model with dropout layer.
        
        Parameters:
            img_size (int): the size of input images (img_size, img_size).
            dropout (float): the drop out rate for the dropout layer. Must be less than 1.

        Returns:
            model (class): the EfficientNet-B2 model with dropout layer.
        """

        base_model = efn.EfficientNetB2(weights=None, include_top=False, 
                                 input_shape = (img_size,img_size,3))
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout)(x)
        predictions = layers.Dense(1, activation='sigmoid', name='last')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(self.weights)
        return model


# Functions for image preprocessing and plotting and saving results

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class imgUtil():
    def _init_(self, img, img_size):
        self.img = img
        self.img_size = img_size
        
    def preprocess(self, std, mean):
        self.img /= 255
        centered = np.subtract(self.img, mean)
        standardized = np.divide(centered, std)
        return self.img
        return standardized
    
    def data_gen(self, rotation, h_shift, w_shift):
        train_datagen = ImageDataGenerator(
                                   preprocessing_function=self.preprocess,
                                   height_shift_range=h_shift,
                                   width_shift_range=w_shift,
                                   rotation_range=rotation,
                                   zoom_range=0.05,
                                   brightness_range=[0.9,1.1],
                                   fill_mode='constant',
                                   horizontal_flip=True
                                                                            )
        test_datagen = ImageDataGenerator(
                                          preprocessing_function=self.preprocess
                                                                            )
        
        return train_datagen, test_datagen
    
    def generator(self, batch_size, train, test, train_dir, test_dir):
        train_generator = train.flow_from_directory(train_dir, 
                                                    target_size = (self.img_size, self.img_size), 
                                                    class_mode = 'binary', 
                                                    color_mode ='rgb', 
                                                    batch_size = batch_size,
                                                    interpolation = 'lanczos'
                                                    )
        validation_generator = test.flow_from_directory(test_dir, 
                                                        target_size = (self.img_size, self.img_size), 
                                                        class_mode = 'binary',
                                                        color_mode = 'rgb', 
                                                        batch_size = batch_size,
                                                        interpolation ='lanczos'
                                                        )
        return train_generator, validation_generator
    
    def plot_save(self, history, save_dir, exp_name):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(save_dir + 'result_')
        plt.show()
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(save_dir + 'save_plots_initial/{}/loss'.format(exp_name))
        plt.show()
        
        plt.plot(history.history['auc'])
        plt.plot(history.history['val_auc'])
        plt.title('Model auc')
        plt.ylabel('AUC')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(save_dir + 'save_plots_initial/{}/auc_hist'.format(exp_name))
        plt.show()
        
        plt.plot(history.history['precision'])
        plt.plot(history.history['val_precision'])
        plt.title('Model precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(save_dir + 'save_plots_initial/{}/prec'.format(exp_name))
        plt.show()
        
        plt.plot(history.history['recall'])
        plt.plot(history.history['val_recall'])
        plt.title('Model recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(save_dir + 'save_plots_initial/{}/recall'.format(exp_name))
        plt.show()
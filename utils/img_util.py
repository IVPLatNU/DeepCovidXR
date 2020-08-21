# Functions for image preprocessing and plotting and saving results

import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing import image
from vis.visualization import visualize_cam
from vis.utils import utils
from PIL import Image
from sklearn.metrics import roc_curve, auc

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


class imgUtils:
    def __init__(self, img_size):
        self.img_size = img_size

    @staticmethod
    def preprocess(img):
        img /= 255
        centered = np.subtract(img, imagenet_mean)
        standardized = np.divide(centered, imagenet_std)
        return standardized

    def resizeImg(self, input_img_path):
        im = Image.open(input_img_path)
        im = im.resize((self.img_size, self.img_size), resample=Image.LANCZOS)
        return im

    def proc_img(self, img_path):
        img = image.load_img(img_path, target_size=(self.img_size, self.img_size),
                             color_mode='rgb', interpolation='lanczos')
        img_array = np.asarray(img, dtype='uint8')
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def dataGen(self, rotation, h_shift, w_shift):
        TTA_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess,
            height_shift_range=h_shift,
            width_shift_range=w_shift,
            rotation_range=rotation,
            zoom_range=0.05,
            brightness_range=[0.8, 1.2],
            fill_mode='constant',
            horizontal_flip=True
        )
        nTTA_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess
        )

        return TTA_datagen, nTTA_datagen

    def generator(self, batch_size, train, test, train_dir, test_dir):
        train_generator = train.flow_from_directory(train_dir,
                                                    target_size=(self.img_size, self.img_size),
                                                    class_mode='binary',
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    interpolation='lanczos'
                                                    )
        test_generator = test.flow_from_directory(test_dir,
                                                  target_size=(self.img_size, self.img_size),
                                                  class_mode='binary',
                                                  color_mode='rgb',
                                                  batch_size=batch_size,
                                                  interpolation='lanczos'
                                                  )
        return train_generator, test_generator

    # The train and test generators used for ensembling
    def testgenerator_from_folder(self, batch_size, test, data_dir):
        test_generator = test.flow_from_directory(data_dir,
                                                  target_size=(self.img_size, self.img_size),
                                                  class_mode='binary',
                                                  color_mode='rgb',
                                                  batch_size=batch_size,
                                                  interpolation='lanczos',
                                                  shuffle=False
                                                  )
        return test_generator

    def testgenerator_from_dataframe(self, batch_size, test, dataframe, class_mode):
        test_generator = test.flow_from_dataframe(dataframe,
                                                  target_size=(self.img_size, self.img_size),
                                                  class_mode=class_mode,
                                                  color_mode='rgb',
                                                  batch_size=batch_size,
                                                  interpolation='lanczos',
                                                  shuffle=False
                                                  )
        return test_generator

    @staticmethod
    def plot_save(history, save_dir):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(os.path.join(save_dir, 'accuracy'))
        plt.show()
        plt.close()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(os.path.join(save_dir, 'loss'))
        plt.show()
        plt.close()

        plt.plot(history.history['auc'])
        plt.plot(history.history['val_auc'])
        plt.title('Model auc')
        plt.ylabel('AUC')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(os.path.join(save_dir, 'auc'))
        plt.show()
        plt.close()

        plt.plot(history.history['precision'])
        plt.plot(history.history['val_precision'])
        plt.title('Model precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(os.path.join(save_dir, 'precision'))
        plt.show()
        plt.close()

        plt.plot(history.history['recall'])
        plt.plot(history.history['val_recall'])
        plt.title('Model recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(os.path.join(save_dir, 'recall'))
        plt.show()
        plt.close()

    # Generates confusion matrix
    @staticmethod
    def confusionMatrix(classes, ensemble_pred_round, result_path):
        conf_matrix = confusion_matrix(classes, ensemble_pred_round)
        target_names = ['COVID-Neg', 'COVID-Pos']
        disp = ConfusionMatrixDisplay(conf_matrix, display_labels=target_names)
        disp = disp.plot(cmap='Blues', values_format='.0f')
        plt.show()
        plt.savefig(os.path.join(result_path, 'ConfusionMatrix.png'))
        print('Classification Report')
        print(classification_report(classes, ensemble_pred_round, target_names=target_names))
        plt.close()
        return conf_matrix

    @staticmethod
    def ROCcurve(classes, ensemble_pred, result_path):
        fpr, tpr, thresholds = roc_curve(classes, ensemble_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(roc_auc))
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(result_path, 'ROC_Curve.png'))
        plt.close()

    # Generates gradcam images
    def gradCAM(self, input_img, model_list):
        pred_list = []
        layer_name_list = ['conv5_block3_3_bn', 'block14_sepconv2_bn', 'bn',
                           'mixed10', 'conv_7b_bn', 'top_bn']
        i = 0
        for model in model_list:
            visualization = visualize_cam(model,
                                          layer_idx=utils.find_layer_idx(model, 'last'),
                                          filter_indices=0,
                                          seed_input=input_img,
                                          penultimate_layer_idx=utils.find_layer_idx(model, layer_name_list[i]),
                                          backprop_modifier=None)
            pred_list.append(visualization)
            i += 1
        visualization = np.mean(pred_list, axis=0)
        return visualization

# Set checkpoint, early stopping and reduce learning rate on plataeu
import os
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from covid_models import DenseNet, ResNet, XceptionNet, EfficientNet, InceptionNet, InceptionResNet 


class trainFeatures():
    def getModel(self, model_name, img_size, weight):
        if model_name == 'ResNet-50':
            resnet = ResNet(weight)
            base = resnet.buildTunerModel(img_size)
            model = resnet.buildBaseModel(img_size)
            freeze_model = resnet.freeze(model)

        elif model_name == 'Xception':
            xception = XceptionNet(weight)
            base = xception.buildTunerModel(img_size)
            model = xception.buildBaseModel(img_size)
            freeze_model = xception.freeze(model)

        elif model_name == 'DenseNet-121':
            dense = DenseNet(weight)
            base = dense.buildTunerModel(img_size)
            model = dense.buildBaseModel(img_size)
            freeze_model = dense.freeze(model)

        elif model_name == 'Inception-V3':
            inception = InceptionNet(weight)
            base = inception.buildTunerModel(img_size)
            model = inception.buildBaseModel(img_size)
            freeze_model = inception.freeze(model)

        elif model_name == 'Inception-ResNet-V2':
            inceptionres = InceptionResNet(weight)
            base = inceptionres.buildTunerModel(img_size)
            model = inceptionres.buildBaseModel(img_size)
            freeze_model = inceptionres.freeze(model)

        elif model_name == 'EfficientNet-B2':
            efficient = EfficientNet(weight)
            base = efficient.buildTunerModel(img_size)
            model = efficient.buildBaseModel(img_size)
            freeze_model = efficient.freeze(model)

        return freeze_model, model, base
    
    def getNihModel(self, model_name, img_size, weight, label_len):
        if model_name == 'ResNet-50':
            resnet = ResNet(weight)
            model = resnet.buildNihModel(img_size, label_len)
            freeze_model = resnet.freeze(model)

        elif model_name == 'Xception':
            xception = XceptionNet(weight)
            model = xception.buildNihModel(img_size, label_len)
            freeze_model = xception.freeze(model)

        elif model_name == 'DenseNet-121':
            dense = DenseNet(weight)
            model = dense.buildNihModel(img_size, label_len)
            freeze_model = dense.freeze(model)

        elif model_name == 'Inception-V3':
            inception = InceptionNet(weight)
            model = inception.buildNihModel(img_size, label_len)
            freeze_model = inception.freeze(model)

        elif model_name == 'Inception-ResNet-V2':
            inceptionres = InceptionResNet(weight)
            model = inceptionres.buildNihModel(img_size, label_len)
            freeze_model = inceptionres.freeze(model)

        elif model_name == 'EfficientNet-B2':
            efficient = EfficientNet(weight)
            model = efficient.buildNihModel(img_size, label_len)
            freeze_model = efficient.freeze(model)

        return freeze_model, model
    
    
    def getDropoutModel(self, model_name, img_size, weight, dropout):
        if model_name == 'ResNet-50':
            resnet = ResNet(weight)
            drop_model = resnet.buildDropModel(img_size, dropout)
            drop_model = resnet.freeze(drop_model)

        elif model_name == 'Xception':
            xception = XceptionNet(weight)
            drop_model = xception.buildDropModel(img_size, dropout)
            drop_model = xception.freeze(drop_model)

        elif model_name == 'DenseNet-121':
            dense = DenseNet(weight)
            drop_model = dense.buildDropModel(img_size, dropout)
            drop_model = dense.freeze(drop_model)

        elif model_name == 'Inception-V3':
            inception = InceptionNet(weight)
            drop_model = inception.buildDropModel(img_size, dropout)
            drop_model = inception.freeze(drop_model)

        elif model_name == 'Inception-ResNet-V2':
            inceptionres = InceptionResNet(weight)
            drop_model = inceptionres.buildDropModel(img_size, dropout)
            drop_model = inceptionres.freeze(drop_model)

        elif model_name == 'EfficientNet-B2':
            efficient = EfficientNet(weight)
            drop_model = efficient.buildDropModel(img_size, dropout)
            drop_model = efficient.freeze(drop_model)

        return drop_model
    
    def getAllModel(self, img_size, weight_dir, crop_stat):
        res_weight = os.path.join(weight_dir, 
                                  'ResNet50_{size}_up_{crop}.h5'.format(size = img_size, crop = crop_stat))
        xception_weight = os.path.join(weight_dir,
                                       'Xception_{size}_up_{crop}.h5'.format(size = img_size, crop = crop_stat))
        dense_weight = os.path.join(weight_dir,
                                    'DenseNet_{size}_up_{crop}.h5'.format(size = img_size, crop = crop_stat))
        inception_weight = os.path.join(weight_dir, 
                                        'Inception_{size}_up_{crop}.h5'.format(size = img_size, crop = crop_stat))
        inceptionres_weight = os.path.join(weight_dir, 
                                           'InceptionResNet_{size}_up_{crop}.h5'.format(size = img_size, crop = crop_stat))
        efficient_weight = os.path.join(weight_dir, 
                                        'EfficientNet_{size}_up_{crop}.h5'.format(size = img_size, crop = crop_stat))
        
        resnet = ResNet(res_weight)
        res_model = resnet.buildBaseModel(img_size)
        xception = XceptionNet(xception_weight)
        xception_model = xception.buildBaseModel(img_size)
        dense = DenseNet(dense_weight)
        dense_model = dense.buildBaseModel(img_size)
        inception = InceptionNet(inception_weight)
        inception_model = inception.buildBaseModel(img_size)
        inceptionres = InceptionResNet(inceptionres_weight)
        inceptionres_model = inceptionres.buildBaseModel(img_size)
        efficient = EfficientNet(efficient_weight)
        efficient_model = efficient.buildBaseModel(img_size)
        
        return res_model, xception_model, dense_model, inception_model, inceptionres_model, efficient_model

    def getAllModelFast(self, img_size, weight_dir, crop_stat):
        res_weight = os.path.join(weight_dir,
                                  f'ResNet50_{img_size}_up_{crop_stat}.h5')
        xception_weight = os.path.join(weight_dir,
                                       f'Xception_{img_size}_up_{crop_stat}.h5')
        dense_weight = os.path.join(weight_dir,
                                    f'DenseNet_{img_size}_up_{crop_stat}.h5')
        inception_weight = os.path.join(weight_dir,
                                        f'Inception_{img_size}_up_{crop_stat}.h5')
        inceptionres_weight = os.path.join(weight_dir,
                                           f'InceptionResNet_{img_size}_up_{crop_stat}.h5')
        efficient_weight = os.path.join(weight_dir,
                                        f'EfficientNet_{img_size}_up_{crop_stat}.h5')

        res_model = ResNet(res_weight).buildBaseModelFast()
        xception_model = XceptionNet(xception_weight).buildBaseModelFast()
        dense_model = DenseNet(dense_weight).buildBaseModelFast()
        inception_model = InceptionNet(inception_weight).buildBaseModelFast()
        inceptionres_model = InceptionResNet(inceptionres_weight).buildBaseModelFast()
        efficient_model = EfficientNet(efficient_weight).buildBaseModelFast()

        return res_model, xception_model, dense_model, inception_model, inceptionres_model, efficient_model
    
    def getAllDropModel(self, img_size, weight_dir, crop_stat, dropout):
        res_weight = os.path.join(weight_dir, 
                                  'ResNet50_{size}_up_{crop}.h5'.format(size = img_size, crop = crop_stat))
        xception_weight = os.path.join(weight_dir,
                                       'Xception_{size}_up_{crop}.h5'.format(size = img_size, crop = crop_stat))
        dense_weight = os.path.join(weight_dir,
                                    'DenseNet_{size}_up_{crop}.h5'.format(size = img_size, crop = crop_stat))
        inception_weight = os.path.join(weight_dir, 
                                        'Inception_{size}_up_{crop}.h5'.format(size = img_size, crop = crop_stat))
        inceptionres_weight = os.path.join(weight_dir, 
                                           'InceptionResNet_{size}_up_{crop}.h5'.format(size = img_size, crop = crop_stat))
        efficient_weight = os.path.join(weight_dir, 
                                        'EfficientNet_{size}_up_{crop}.h5'.format(size = img_size, crop = crop_stat))
        
        resnet = ResNet(res_weight)
        res_model = resnet.buildDropModel(img_size, dropout)
        xception = XceptionNet(xception_weight)
        xception_model = xception.buildDropModel(img_size, dropout)
        dense = DenseNet(dense_weight)
        dense_model = dense.buildDropModel(img_size, dropout)
        inception = InceptionNet(inception_weight)
        inception_model = inception.buildDropModel(img_size, dropout)
        inceptionres = InceptionResNet(inceptionres_weight)
        inceptionres_model = inceptionres.buildDropModel(img_size, dropout)
        efficient = EfficientNet(efficient_weight)
        efficient_model = efficient.buildDropModel(img_size, dropout)
        
        return res_model, xception_model, dense_model, inception_model, inceptionres_model, efficient_model

    def setCP(self, monitor, model_path):
        checkpoint = ModelCheckpoint(
                                        filepath = model_path,
                                        verbose=1,
                                        monitor=monitor,
                                        save_best_only=True,
                                        mode='max')
        return checkpoint
    
    def setES(self, monitor, patience, min_delta):
        es = EarlyStopping(monitor=monitor, 
                   verbose=1, 
                   patience=patience, 
                   min_delta=min_delta, 
                   mode='max')
        return es
    
    def setRLP(self, monitor, factor, patience):
        rlr = ReduceLROnPlateau(monitor=monitor,
                        mode='max',
                        factor=factor,
                        patience=patience)
        return rlr
        
    def NIHgenerator(self, model, batch_size, train_gen, val_gen, epochs, cp, rlr, es):
        model.fit_generator(train_gen,
              steps_per_epoch=len(train_gen.classes)/batch_size,
              validation_data=val_gen,
              validation_steps=len(val_gen.classes)/batch_size,
              epochs=epochs,
              callbacks=[cp, rlr, es])
     
    def load(self, model, weights):
        model.load_weights(weights)
        return model
    
    def unfreeze(self, model):
        for layer in model.layers[0:]:
            layer.trainable = True
        return model
        
    def generator(self, model, train_gen, val_gen, epochs, cp, rlr, es):
        history = model.fit_generator(train_gen, 
                             epochs = epochs, 
                             validation_data=val_gen, 
                             callbacks = [es, cp, rlr])
        return history
    
    def compileModel(self, model, lr, momentum, nestrov):
        model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(learning_rate=lr, 
                                             momentum=momentum,
                                             nesterov=nestrov,
                                             ), 
              metrics=['acc', 
                    tf.keras.metrics.AUC(name='auc'), 
                    tf.keras.metrics.Precision(name='precision'), 
                    tf.keras.metrics.Recall(name='recall')])

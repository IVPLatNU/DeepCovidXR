# Set checkpoint, early stopping and reduce learning rate on plataeu
from tensorflow.keras import optimizers
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from covid_models import hyperModel, DenseNet, ResNet, XceptionNet, EfficientNet, InceptionNet, InceptionResNet 

class trainFeatures():
    def getModel(self, model_name):
        if model_name == 'ResNet-50':
            resnet = ResNet(nih_weight)
            base = resnet.buildTunerModel(img_size)
            model = resnet.buildBaseModel(img_size)
            freeze_model = resnet.freeze(model)

        elif model_name == 'Xception':
            xception = XceptionNet(nih_weight)
            base = xception.buildTunerModel(img_size)
            model = xception.buildBaseModel(img_size)
            freeze_model = xception.freeze(model)

        elif model_name == 'DenseNet-121':
            dense = DenseNet(nih_weight)
            base = dense.buildTunerModel(img_size)
            model = dense.buildBaseModel(img_size)
            freeze_model = dense.freeze(model)

        elif model_name == 'Inception-V3':
            inception = InceptionNet(nih_weight)
            base = inception.buildTunerModel(img_size)
            model = inception.buildBaseModel(img_size)
            freeze_model = inception.freeze(model)

        elif model_name == 'Inception-ResNet-V2':
            inceptionres = InceptionResNet(nih_weight)
            base = inceptionres.buildTunerModel(img_size)
            model = inceptionres.buildBaseModel(img_size)
            freeze_model = inceptionres.freeze(model)

        elif model_name == 'EfficientNet-B2':
            efficient = EfficientNet(nih_weight)
            base = efficient.buildTunerModel(img_size)
            model = efficient.buildBaseModel(img_size)
            model = efficient.freeze(model)

        return freeze_model, model, base
    
    def getDropoutModel(self, model_name, dropout):
        # Train with only pooling layers
        if model_name == 'ResNet-50':
            resnet = ResNet(nih_weight)
            drop_model = resnet.buildDropModel(dropout)
            drop_model = resnet.freeze(drop_model)

        elif model_name == 'Xception':
            xception = XceptionNet(nih_weight)
            drop_model = xception.buildDropModel(dropout)
            drop_model = xception.freeze(drop_model)

        elif model_name == 'DenseNet-121':
            dense = DenseNet(nih_weight)
            drop_model = dense.buildDropModel(dropout)
            drop_model = dense.freeze(drop_model)

        elif model_name == 'Inception-V3':
            inception = InceptionNet(nih_weight)
            drop_model = inception.buildDropModel(dropout)
            drop_model = inception.freeze(drop_model)

        elif model_name == 'Inception-ResNet-V2':
            inceptionres = InceptionResNet(nih_weight)
            drop_model = inceptionres.buildDropModel(dropout)
            drop_model = inceptionres.freeze(drop_model)

        elif model_name == 'EfficientNet-B2':
            efficient = EfficientNet(nih_weight)
            drop_model = efficient.buildDropModel(dropout)
            drop_model = efficient.freeze(drop_model)

        return drop_model

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
        
    
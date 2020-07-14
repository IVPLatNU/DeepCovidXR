# Set checkpoint, early stopping and reduce learning rate on plataeu
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#import tensorflow as tf

class trainFeatures():
        
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
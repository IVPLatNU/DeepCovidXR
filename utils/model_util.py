# Set checkpoint, early stopping and reduce learning rate on plataeu
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

class trainFeatures():
        
    def setCP(self, model_path, monitor):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
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
        
    def generator(self, model, batch_size, train_gen, val_gen, epochs, cp, rlr, es):
        model.fit_generator(train_gen,
              steps_per_epoch=len(train_gen.classes)/batch_size,
              validation_data=valid_gen,
              validation_steps=len(valid_gen.classes)/batch_size,
              epochs=epochs,
              callbacks=[cp, rlr, es])
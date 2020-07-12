# Pretrain a given model on NIH dataset

from utils import nihUtils, imgUtils, trainFeatures
from covid_models import DenseNet

img_size = 331
batch_size = 16
rotation_range = 20
height_shift = 0.05
width_shift = 0.05

model_save_path = 'nih_saved_weights.h5'

nih = nihUtils()
nih_path = nih.nihDir()
nih.nihDownload()

nih_img_path = nih_path
nih_csv_path = nih_path
csv_name = nih_csv_path + 'NIH_Data_Entry.csv'

train_df, val_df, labels = nih.nihSplit(csv_name, nih_img_path, batch_size)

img_proc = imgUtils(img_size)
train_idg, val_idg = img_proc.dataGen(rotation_range, height_shift, width_shift)

train_generator, val_generator = nih.nihGenerator(batch_size, train_idg, val_idg, train_df, val_df, labels)

# Train a given model on NIH dataset
lr = 0.001
momentum = 0.9
nestrov = True

patience_rlr = 2
patience_es = 10
factor = 0.1
min_delta = 0.001
monitor = 'val_auc'

dense = DenseNet('imagenet')
model = dense.buildBaseModel(img_size)
dense.compileModel(model, lr, momentum, nestrov)

features = trainFeatures()
rlp = features.setRLP(monitor, factor, patience_rlr)
es = features.setES(monitor, patience_es, min_delta)
cp = features.setCP(monitor, model_save_path, monitor)

epochs = 50
featuers.generator(model, batch_size, train_generator, val_generator, epochs, cp, rlp, es)







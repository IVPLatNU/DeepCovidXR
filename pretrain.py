# Pretrain a given model on NIH dataset

from utils import nihUtils

nih = nihUtils()
nih_path = nih.nihDir()
nih.nihDownload()

nih_img_path = nih_path
nih_csv_path = nih_path
csv_name = nih_csv_path + 'NIH_Data_Entry.csv'

train_df, val_df = nih.nihSplit(csv_name, nih_img_path)





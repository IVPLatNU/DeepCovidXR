import os
from glob import glob
from itertools import chain
from sklearn.model_selection import train_test_split
import pandas as pd


def nihSplit(self, csv_path, img_path):
    df = pd.read_csv(csv_path)
    data_image_paths = {os.path.basename(x): x for x in glob(os.path.join(img_path, '*.png'))}
    df['Finding Labels'] = df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
    df['path'] = df['Image Index'].map(data_image_paths.get)
    
    labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    labels = [x for x in labels if len(x) > 0]
    
    for label in labels:
        if len(label) > 1:
            df[label] = df['Finding Labels'].map(lambda finding: 1.0 if label in finding else 0.0)
    
    labels = [label for label in labels if df[label].sum() > 1000]
    
    
    train_df, valid_df = train_test_split(df, test_size=0.20, random_state=2018, 
                                          stratify=df['Finding Labels'].map(lambda x: x[:4]))
    
    
    train_df['labels'] = train_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
    valid_df['labels'] = valid_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
    
    return train_df, valid_df

                
import os
from glob import glob
import numpy as np
from itertools import chain
import tarfile
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import urllib

class nihUtils():
    # Create a direcotry for NIH dataset
    def nihDir(self):
        base = os.getcwd()
        nih_dir = base + '\\NIH\\'
        if not os.path.isdir(nih_dir):
            os.mkdir(nih_dir)
        return nih_dir
    
    # Download NIH dataset
    def nihDownload(self, nih_dir):
        links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
        ]
        
        for idx, link in enumerate(links):
            fn = nih_dir + 'images_%02d.tar.gz' % (idx+1)
            print ('downloading', fn, '...')
            urllib.request.urlretrieve(link, fn)  # download the zip file
            tar = tarfile.open(fn)
            tar.extractall()
            tar.close()
        print ("NIH dataset download and unzip complete. ")
        
    
    # Split NIH dataset into train set and validation set
    def nihSplit(self, csv_name, img_path):
        df = pd.read_csv(csv_name)
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
        
        return train_df, valid_df, labels
    
    def nihGenerator(self, batch_size, train, val, train_df, valid_df, labels):
        train_gen = train_idg.flow_from_dataframe(dataframe=train_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='labels',
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             classes=labels,
                                             target_size=(self.image_size, self.image_size))


        valid_gen = val_idg.flow_from_dataframe(dataframe=val_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='labels',
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             classes=labels,
                                             target_size=(self.image_size, self.image_size))
        return train_gen, valid_gen
   
        



                
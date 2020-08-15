import os
import io
import sys
import time
from glob import glob
import numpy as np
from itertools import chain
import tarfile
from sklearn.model_selection import train_test_split
import pandas as pd
import urllib

class nihUtils():
    # Create a direcotry for NIH dataset
    def createDir(self, nih_dir):
        create = False
        if not os.path.isdir(nih_dir):
            create = True
            os.mkdir(nih_dir)
        return nih_dir, create
    
    def reporthook(self, count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * (int(duration)+1)))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                        (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()    
        
    # Class for printing exrtaction progress
    def get_file_progress_file_object_class(self, on_progress):
        class FileProgressFileObject(tarfile.ExFileObject):
            def read(self, size, *args):
                on_progress(self.name, self.position, self.size)
                return tarfile.ExFileObject.read(self, size, *args)
        return FileProgressFileObject
    
    def on_progress(self, filename, position, total_size):
        print("%s: %d of %s" %(filename, position, total_size), end='\r', flush = True)
        
    class ProgressFileObject(io.FileIO):
        def __init__(self, path, *args, **kwargs):
            self._total_size = os.path.getsize(path)
            io.FileIO.__init__(self, path, *args, **kwargs)
    
        def read(self, size):
            print("Overall process: %d of %d" %(self.tell(), self._total_size), end = '\r', flush = True)
            return io.FileIO.read(self, size)

     
    
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
            fn = os.path.join(nih_dir, 'images_%02d.tar.gz' % (idx+1))
            if not os.path.exists(fn):
                print ('downloading', fn, '...')
                urllib.request.urlretrieve(link, fn, self.reporthook)  # download the zip file
            tarfile.TarFile.fileobject = self.get_file_progress_file_object_class(self.on_progress)
            tar = tarfile.open(fileobj=self.ProgressFileObject(fn))
            print('extracting', fn, '...')
            for member in tar.getmembers():
                if member.isreg():  # skip if the TarInfo is not files
                    member.name = os.path.basename(member.name) # remove the path by reset it
                    tar.extract(member,nih_dir) # extract
            tar.close()
        print ("NIH dataset download and unzip complete. ")
        
    
    # Split NIH dataset into train set and validation set
    def nihSplit(self, csv_name, img_path):
        df = pd.read_csv(csv_name)
        data_image_paths = {os.path.basename(x): x for x in glob(os.path.join(img_path, '*.png'))}
        df['Finding Labels'] = df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
        df['path'] = df['Image Index'].map(data_image_paths.get)
        
        labels = np.unique(list(chain(*df['Finding Labels'].map(
                                            lambda x: x.split('|')).tolist())))
        labels = [x for x in labels if len(x) > 0]
        
        for label in labels:
            if len(label) > 1:
                df[label] = df['Finding Labels'].map(
                    lambda finding: 1.0 if label in finding else 0.0)
        
        labels = [label for label in labels if df[label].sum() > 1000]
        
        
        train_df, valid_df = train_test_split(df, test_size=0.20, random_state=2018, 
                                              stratify=df['Finding Labels'].map(
                                                  lambda x: x[:4]))
        
        
        train_df['labels'] = train_df.apply(
                            lambda x: x['Finding Labels'].split('|'), axis=1)
        valid_df['labels'] = valid_df.apply(
                            lambda x: x['Finding Labels'].split('|'), axis=1)
        label_len = len(labels)
        
        return train_df, valid_df, labels
    
    def nihGenerator(self, image_size, batch_size, train, val, train_df, valid_df, labels):
        train_gen = train.flow_from_dataframe(dataframe=train_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='labels',
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             classes=labels,
                                             target_size=(image_size, image_size))


        valid_gen = val.flow_from_dataframe(dataframe=valid_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='labels',
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             classes=labels,
                                             target_size=(image_size, image_size))
        return train_gen, valid_gen
   
        



                

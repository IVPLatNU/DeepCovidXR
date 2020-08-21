import glob
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_path', type = str, help = 'the directory of the input path')
parser.add_argument('-o','--output_path', type = str, help = 'the directory of the output path')
parser.add_argument('-s', '--size', type = int, help = 'the cropping size')
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
img_size = args.size

if not os.path.exists(output_path):
    os.mkdir(output_path)


LFHTest = []
for file in glob.glob(input_path + '/*'):
#     print(file)
    path, filename = os.path.split(file)
#     print(file)
    filename2, extension = filename.rsplit('.', 1)
    LFHTest.append(filename2)
Test_set = set(LFHTest)
matches = []
for file in glob.glob(input_path + '/*'):
    path, filename = os.path.split(file)
    
    #no crop
    filename2, junk2 = filename.rsplit('.', 2)
    #crop
    if filename2 in Test_set:
        im = Image.open(file)
        im = im.resize((img_size,img_size), resample=Image.LANCZOS)
        im.save(os.path.join(output_path,filename))
        matches.append(filename)
print('Successfully resize {} images to {}*{}'.format( len(matches), img_size, img_size))

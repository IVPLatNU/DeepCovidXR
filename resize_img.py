import glob
from PIL import Image
import os
import argparse

"""Image resizing

This script will perform resizing on a image dataset.

The result will be saved in new directories.
"""


def resize_images(input_path, output_path, img_size):
    """
    This function resizes a given set of images and save the image in the given
    path.
    
    Parameters:
        input_path (string): the path to the image dataset to be resized.
        output_path (string): the path to teh directory where the resized images 
        will be saved.
        img_size (int): the size of the images to be resized to.
        
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    files = glob.glob(os.path.join(input_path, '*'))
    for file in files:
        filename = os.path.basename(file)
        im = Image.open(file)
        im = im.resize((img_size, img_size), resample=Image.LANCZOS)
        im.save(os.path.join(output_path, filename))
    print('Successfully resized {} images to {}*{}'.format(len(files), img_size, img_size))


if __name__ == '__main__':
    """
    The main function takes user input and resizes a image dataset.
    
    The user input variables include the path to the image dataset, the path 
    where the resized images will be saved and the size of the images to be resized
    to. 

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, help='the directory of the input path')
    parser.add_argument('-o', '--output_path', type=str, help='the directory of the output path')
    parser.add_argument('-s', '--size', type=int, help='the cropping size')
    args = parser.parse_args()
    input_path = os.path.normpath(args.input_path)
    output_path = os.path.normpath(args.output_path)
    img_size = args.size

    resize_images(input_path, output_path, img_size)

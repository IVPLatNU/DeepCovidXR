import glob
from PIL import Image
import os
import argparse


def resize_images(input_path, output_path, img_size):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    files = glob.glob(os.path.join(input_path, '*'))
    for file in files:
        filename = os.path.basename(file)
        im = Image.open(file)
        im = im.resize((img_size, img_size), resample=Image.LANCZOS)
        im.save(os.path.join(output_path, filename))
    print('Successfully resize {} images to {}*{}'.format(len(files), img_size, img_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, help='the directory of the input path')
    parser.add_argument('-o', '--output_path', type=str, help='the directory of the output path')
    parser.add_argument('-s', '--size', type=int, help='the cropping size')
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    img_size = args.size

    resize_images(input_path, output_path, img_size)

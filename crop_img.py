import numpy as np
from skimage import io, color, exposure, img_as_float, transform, util
from matplotlib import pyplot as plt
import pathlib
import cv2
import multiprocessing
import time
import argparse
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os

def load_CXR_from_list(filelist, im_shape):
    X = np.zeros((len(filelist), im_shape[0], im_shape[1], 1))
    resized_raw = np.zeros((len(filelist), im_shape[0], im_shape[1]))
    raw_images = []
    
    for k, file in enumerate(filelist):            
        if file.suffix.lower() in ['.jpg', '.png', '.jpeg'] :
            print('loading ' + file.name)
            img = img_as_float(io.imread(file, as_gray = True))
            raw_images.append(img)

            img = transform.resize(img, im_shape)
            resized_raw[k, ...] = img
            
            img = exposure.equalize_hist(img)
            img = np.expand_dims(img, -1)
            X[k, ...] = img

    # X = np.array(X)
    # resized_raw = np.array(resized_raw)
    X -= X.mean()
    X /= X.std()

    print ('### Dataset loaded')
    print ('X shape ={} \t raw_resized shape = {}'.format(X.shape, resized_raw.shape))
    print ('\tX:{:.1f}-{:.1f}\n'.format(X.min(), X.max()))
    print ('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))

    return X, resized_raw, raw_images

def masked(img, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    color_mask[..., 2] = mask / 255
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def draw_spine(img, spine_pos):
    if len(img.shape) == 2:
        img_color = np.dstack((img, img, img))
    elif len(img.shape) == 3 and img.shape[2] == 1:
        squeezed = np.squeeze(img)
        img_color = np.dstack((squeezed, squeezed, squeezed))
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img_color = np.copy(img)
    else:
        raise ValueError('Bad dimension of img :' + str(img.shape))
    
    cv2.rectangle(img_color, (spine_pos, 0), (spine_pos, img.shape[0]), color = (0.8, 0 , 0), thickness = int(round(max(img.shape) * 0.02)))

    return img_color

def draw_bbox(img, bbox):
    '''
    input img, and bounding box
    return a color RGB image (img.shape, 3) with bounding box drawn
    original img is not changed.
    '''


    if len(img.shape) == 2:
        img_color = np.dstack((img, img, img))
    elif len(img.shape) == 3 and img.shape[2] == 1:
        squeezed = np.squeeze(img)
        img_color = np.dstack((squeezed, squeezed, squeezed))
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img_color = np.copy(img)
    else:
        raise ValueError('Bad dimension of img :' + str(img.shape))
    
    if not (bbox is None):    
        left, top, right, bottom = bbox
        cv2.rectangle(img_color, (left, top), (right, bottom), color = (0, 0.8, 0), thickness = int(round(max(img.shape) * 0.01)))
    return img_color


def join_path_from_list(cur_path, path_list):
    for folder in path_list:
        cur_path = cur_path.joinpath(folder)
    return cur_path

def change_first_folder(data_path, attached_str):
    to_return = data_path.copy()
    to_return[0] = to_return[0] + attached_str
    return to_return

def select_spine(img):
    sumpix0 = np.sum(img, axis = 0)
    max_r2 = np.int_(len(sumpix0) / 3) + np.argmax(sumpix0[ np.int_(len(sumpix0) / 3): np.int_(len(sumpix0)* 2 / 3)])
    return max_r2

def mirror(spine_pos, pos):
    if pos < spine_pos:
        return spine_pos + (spine_pos - pos)
    else:
        return spine_pos - (pos - spine_pos)

def left_right(label_map, label, spine_pos):
    left_chunk_size = np.sum(label_map[:, 0 : spine_pos] == label)
    right_chunk_size = np.sum(label_map[:, spine_pos + 1 :] == label)
    if left_chunk_size > right_chunk_size:
        return 'left'
    elif left_chunk_size < right_chunk_size:
        return 'right'
    else:
        return 'mid'

def select_lung(pred, resized_raw, cut_thresh, debug, filename,out_pad_size, k_size = 5):
    opened = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel = np.ones((k_size, k_size)))

    cnt, label_map, stats, centriods = cv2.connectedComponentsWithStats(opened)

    # index sorted by area, from large to small, first one is the background
    idx_sorted = np.argsort(stats[:, cv2.CC_STAT_AREA])[::-1]

    stats = stats[idx_sorted]
    # remove small connected region
    if debug:
        print(stats)
    stats = stats[stats[:, cv2.CC_STAT_AREA] > cut_thresh * np.prod(pred.shape)]

    # only save the largest two or
    denoised = np.zeros(opened.shape, dtype = np.uint8)
    for i in range(1, min(stats.shape[0], 3)):
        denoised[label_map == idx_sorted[i]] = 255

    spine_pos = select_spine(resized_raw)
    
    if stats.shape[0] < 3:
        if stats.shape[0] == 1:
            print(filename + ' No large enough area Detected!!!')
            return denoised, None, spine_pos
        else:
            print(filename + ' Single Lung Detected !!!')
            top = stats[1, cv2.CC_STAT_TOP]
            bottom = stats[1, cv2.CC_STAT_TOP] + stats[1, cv2.CC_STAT_HEIGHT]
            left = stats[1, cv2.CC_STAT_LEFT]
            right = stats[1, cv2.CC_STAT_LEFT] + stats[1, cv2.CC_STAT_WIDTH]

            left_mirror = mirror(spine_pos, left)
            right_mirror = mirror(spine_pos, right)

            left = min(left, right_mirror)
            right = max(right, left_mirror)
            
    else:
        left = min(stats[1, cv2.CC_STAT_LEFT], stats[2, cv2.CC_STAT_LEFT])
        top = min(stats[1, cv2.CC_STAT_TOP], stats[2, cv2.CC_STAT_TOP])
        right = max(
            stats[1, cv2.CC_STAT_LEFT] + stats[1, cv2.CC_STAT_WIDTH],
            stats[2, cv2.CC_STAT_LEFT] + stats[2, cv2.CC_STAT_WIDTH]
        )
        bottom = max(
            stats[1, cv2.CC_STAT_TOP] + stats[1, cv2.CC_STAT_HEIGHT],
            stats[2, cv2.CC_STAT_TOP] + stats[2, cv2.CC_STAT_HEIGHT]
        )
        
        chunk1_side = left_right(label_map, 1, spine_pos)
        chunk2_side = left_right(label_map, 2, spine_pos)
        # print('chunk1 on' + chunk1_side + ' chunk2 on ' + chunk2_side)
        if chunk1_side == chunk2_side:
            print(filename + ' two chunks on the same side!!!')
            left_mirror = mirror(spine_pos, left)
            right_mirror = mirror(spine_pos, right)

            left = min(left, right_mirror)
            right = max(right, left_mirror)
    
    bbox = np.array([left, top, right, bottom])

    bbox = out_pad(bbox, denoised.shape, out_pad_size)
    # boxed = cv2.rectangle(denoised, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color = 255, thickness=3)
    
    # return denoised, bbox, denoised_no_bbox, raw_bbox
    return denoised, bbox, spine_pos

def out_pad(bbox_in, shape, out_pad_size):
    left, top, right, bottom = bbox_in

    left = max(0, left - out_pad_size)
    # right = min(shape[1] - 1, right + out_pad_size)
    right = min(shape[1] , right + out_pad_size)
    top = max(0, top - out_pad_size)
    # bottom = min(shape[0] - 1, bottom + out_pad_size)
    bottom = min(shape[0], bottom + out_pad_size)
    bbox_padded = np.array([left, top, right, bottom])

    return bbox_padded

def square_helper(start, finish, expand1, expand2, size_limit):
    new_start = max(0, start - expand1)
    expand1_rem = expand1 - (start - new_start)
    new_finish = min(size_limit , finish + expand2)
    expand2_rem = expand2 - (new_finish - finish)

    # print('expand1_rem = ', expand1_rem, ' expand2_rem = ', expand2_rem)
    if expand1_rem > 0 and expand2_rem == 0:
        new_finish = min(size_limit, new_finish + expand1_rem)
    elif expand1_rem == 0 and expand2_rem > 0:
        new_start = max(0, new_start - expand2_rem)
    
    return new_start, new_finish

def square_bbox(img_shape, raw_bbox):
    if raw_bbox is None:
        return None
    # img_shape = denoised_no_bbox.shape
    left, top, right, bottom = raw_bbox
    width = right - left
    height = bottom - top
    center = [round((left + right) / 2), round((top + bottom) / 2)]

    diff = abs(width - height)
    expand1 = diff // 2
    expand2 = diff - expand1

    sqaured_bbox = np.copy(raw_bbox)
    # print('expand1 = ', expand1, ' expand2 = ', expand2)
    if width > height:
        new_top, new_bottom = square_helper(top, bottom, expand1, expand2, img_shape[0])
        sqaured_bbox = np.array([left, new_top, right, new_bottom])
    elif width < height:
        new_left, new_right = square_helper(left, right, expand1, expand2, img_shape[1])
        sqaured_bbox = np.array([new_left, top, new_right, bottom])

    # print('original bounding box:' + str(raw_bbox))
    # print('squared bounding box:' + str(sqaured_bbox))
    return sqaured_bbox

def bbox_mask_and_crop(raw_img, bbox):
    '''
    return the cropped image, bounding box mask, and the bounding box itself.
    '''
    if bbox is None:
        return raw_img, bbox

    left, top, right, bottom = bbox

    cropped_img = raw_img[
        top : bottom,
        left : right
    ]

    return cropped_img, bbox

def square_crop(raw_img, raw_bbox):
    if raw_bbox is None:
        return raw_img, raw_bbox
    sqaured_bbox = square_bbox(raw_img.shape, raw_bbox)
    
    return bbox_mask_and_crop(raw_img, sqaured_bbox)

def crop_lung(raw_img, cur_shape, bbox):
    if bbox is None:
        return raw_img, None
    if len(bbox) != 4:
        raise ValueError('WRONG length of bounding box')

    left, top, right, bottom = bbox
    raw_height = raw_img.shape[0]
    raw_width = raw_img.shape[1]
    cur_height = cur_shape[0]
    cur_width = cur_shape[1]

    # print('Bounding box = {}'.format(bbox))
    # print('raw shape = {}'.format(raw_img.shape))
    # print('cur shape = {}'.format(cur_shape))
    
    lung_top = int(round(top / cur_height * raw_height))
    lung_bottom = int(round(bottom / cur_height * raw_height))
    lung_left = int(round(left / cur_width * raw_width))
    lung_right = int(round(right / cur_width * raw_width))

    # print('lung left = {} right = {} top = {} bottom = {} '.format(lung_left, lung_right, lung_top, lung_bottom))
    lung_img = raw_img[
        lung_top : lung_bottom,
        lung_left : lung_right
    ]

    # print('lung shape = {}'.format(lung_img.shape))
    raw_bbox = np.array([lung_left, lung_top, lung_right, lung_bottom])

    return bbox_mask_and_crop(raw_img, raw_bbox)

def pretty(filename, char_per_line):
    return '\n'.join(filename[i : i + char_per_line] for i in range(0, len(filename), char_per_line))

def single_img_crop(img, resized_raw_img, raw_img, file_path, UNet, result_folder, 
                        im_shape = (256, 256), cut_thresh = 0.02, out_pad_size = 8, debug_folder = None , debugging = False):
    '''
    Crop out the lung area from CXR for single images\n
    lung prediction based on UNet
    Parameters
    ----------
    img : np array 
        acceptable shape: (n, x, x, 1), (n, x, x), (x, x, 1), (x, x)
        where n is the number of images; x is the input_shape, by default 256
    resized_raw_img : np array
        raw sized image, with shape of (x, x);
        see load_CXR_from_list for details
    raw_img : np array
        original raw image;
        see load_CXR_from_list for details
    UNet: loaded UNet model from https://github.com/imlab-uiip/lung-segmentation-2d
        path to UNet    
    result_folder : preferrebly pathlib object
        path to output
    im_shape : tuple
        specify the input image shape of UNet, by default (256, 256)
    cut_thresh: float
        connected components less than cut_thresh * np.prod(im_shape) will be removed
    out_pad_size: int
        Default to be 8, how many pixels to enlarge the bounding box. 
    debug_folder : preferrebly pathlib object
        path to debug images; if not specified, no debug images will be written to local
    debugging: bool
        Default to be false. If true, will plot debugging images to screen instead of saving to local.
    Returns
    ----------
    lung_img : np array
        cropped lung area (not neccessarily squared)        
    lung_img_squared :
        cropped lung area (squared if possible)
    '''
    # we need (n, x, x, 1) format for input of Unet
    # n is the number of images
    # x is the input shape, by default 256
    if len(img.shape) == 4 and img.shape[1: -1] == im_shape and img.shape[-1] == 1:
        # format (n, x, x, 1)
        pass
    elif len(img.shape) == 2 and img.shape == im_shape:
        # format (x, x)
        img = np.expand_dims(img, axis = (0, -1))
    elif len(img.shape) == 3 and img.shape[:2] == im_shape and img.shape[-1] == 1:
        # format (x, x, 1)
        img = np.expand_dims(img, axis = 0)
    elif len(img.shape) == 3 and img.shape[1:] == im_shape:
        # format (n, x, x)
        img = np.expand_dims(img, axis = -1)
    else:
        raise ValueError('Bad dimension of img :' + str(img.shape))

    if not (debug_folder is None) or debugging:
        fig, axes = plt.subplots(2, 2, figsize = (8, 8))
        fig2, axes2 = plt.subplots(1, 3, figsize = (18, 6))
    
    pred = np.squeeze(UNet.predict(img))
    pr = (pred > 0.5).astype(np.uint8)

    if not file_path == None:
        filename = file_path.stem
        suffix = file_path.suffix
        print('outputting result for ' + filename)

    denoised, raw_bbox, spine_pos = select_lung(pr, resized_raw_img, cut_thresh = cut_thresh, debug = debugging, filename = filename, out_pad_size = out_pad_size)
    # denoised_sqaured, sqaured_bbox = square_bbox(denoised_no_bbox, raw_bbox)

    lung_img, nonSquared_bbox = crop_lung(raw_img, im_shape, raw_bbox)
    lung_img_squared, sqaured_bbox = square_crop(raw_img, nonSquared_bbox)    
    lung_img = util.img_as_ubyte(lung_img)
    lung_img_squared = util.img_as_ubyte(lung_img_squared)
    
    if not (debug_folder is None) or debugging:
        axes[0, 0].set_title(pretty(filename, char_per_line = 20) + '\n'+ '_resized_raw')
        axes[0, 0].imshow(resized_raw_img, cmap='gray')
        axes[0, 0].set_axis_off()

        axes[1, 0].set_title(pretty(filename, char_per_line = 20) + '\n'+ '_rawpred')
        axes[1, 0].imshow(pr ,cmap='gray')
        axes[1, 0].set_axis_off()

        axes[0, 1].set_title(pretty(filename, char_per_line = 20) + '\n'+ '_denoised_pred')
        axes[0, 1].imshow(denoised, cmap='gray')
        axes[0, 1].set_axis_off()

        axes[1, 1].set_title(pretty(filename, char_per_line = 20) + '\n'+ '_denoised_masked')
        area_masked = masked(resized_raw_img, denoised, alpha = 0.6)
        bbox_drawn = draw_bbox(area_masked, raw_bbox)
        spine_drawn = draw_spine(bbox_drawn, spine_pos)
        axes[1, 1].imshow(spine_drawn)
        axes[1, 1].set_axis_off()
        
        fig.tight_layout()

        axes2[0].set_title(pretty(filename, char_per_line = 30) + '\n'+ 'raw_img')
        axes2[0].imshow(raw_img, cmap='gray')
        # axes2[0].set_axis_off()

        axes2[1].set_title(pretty(filename, char_per_line = 30) + '\n'+ 'unsquared_boudning_box' + '\n' + str(nonSquared_bbox))
        axes2[1].imshow(draw_bbox(raw_img, nonSquared_bbox))
        # axes2[1].set_axis_off()

        axes2[2].set_title(pretty(filename, char_per_line = 30) + '\n'+ 'squared_boudning_box'+ '\n' + str(sqaured_bbox))
        axes2[2].imshow(draw_bbox(raw_img, sqaured_bbox))
        
        fig.tight_layout()
        
        if debugging:
            plt.show()
        elif not (debug_folder is None):
            out_path = debug_folder.joinpath(filename + '_debug_resized_scale' + suffix)
            fig.savefig(str(out_path))
            out_path = debug_folder.joinpath(filename + '_debug_rawscale' + suffix)
            fig2.savefig(str(out_path))

    if not debugging:
        if not result_folder == None:
            result_sub = result_folder.joinpath('crop')
            result_sub.mkdir(parents=True, exist_ok=True)
            out_path = result_sub.joinpath(filename + '_crop' + suffix)
            io.imsave(str(out_path), lung_img )
    
            result_sub = result_folder.joinpath('crop_squared')
            result_sub.mkdir(parents=True, exist_ok=True)
            out_path = result_sub.joinpath(filename + '_crop_squared' + suffix)
            io.imsave(str(out_path), lung_img_squared )
    
    if not (debug_folder is None) or debugging:
        plt.close(fig)
        plt.close(fig2)

    return lung_img, lung_img_squared

def lungseg_fromdata(X, resized_raw, raw_images, file_paths, UNet, result_folder, 
                        im_shape = (256, 256), cut_thresh = 0.02, out_pad_size = 8, debug_folder = None ,debugging = False):

    # tf.debugging.set_log_device_placement(True)
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # with tf.device('/GPU:0'):
    n_test = X.shape[0]
    inp_shape = X[0].shape
    UNet = load_model(UNet)
    print('n_test = {}'.format(n_test))
    # For inference standard keras ImageGenerator can be used.
    test_gen = ImageDataGenerator(rescale=1.)
    
    i = 0
    for xx in test_gen.flow(X, batch_size=1, shuffle=False): 
        single_img_crop(
            img = xx, 
            resized_raw_img = resized_raw[i], 
            raw_img = raw_images[i], 
            file_path = file_paths[i], 
            UNet = UNet,
            result_folder = result_folder,
            im_shape = im_shape,
            cut_thresh = cut_thresh,
            out_pad_size = out_pad_size,
            debug_folder = debug_folder,
            debugging = debugging
        )
            
        i += 1
        if i == n_test:
            break
        
    print('Thread done')

def gen_idx(length, k_fold):
    idxs = np.array([length // k_fold] * k_fold)    
    idxs[:length % k_fold] = idxs[:length % k_fold] + 1
    start_points = np.cumsum(idxs)
    start_points = [0] + list(start_points)
    return start_points

def adjust_process_num(length):
    if length < 20:
        k_fold = 1
    elif length < 100:
        k_fold = 4
    elif length < 400:
        k_fold = 8
    elif length < 1000:
        k_fold = 16
    else:
        k_fold = 24

    return k_fold

def lungseg_one_process(result_folder, UNet, filenames, 
                        im_shape = (256, 256), debug_folder = None, cut_thresh = 0.02, out_pad_size = 8, debug = False):
    
    X, resized_raw, raw_images = load_CXR_from_list(filenames, im_shape)
    print('X shape = ', X.shape)  
    lungseg_fromdata(X, resized_raw, raw_images, filenames, UNet, result_folder, 
                        im_shape = im_shape, cut_thresh = cut_thresh, out_pad_size = out_pad_size, debug_folder = debug_folder, debugging = debug)

def singlefolder_lungseg(data_path, result_folder, UNet, debug_folder = None, k_fold = None, cut_thresh = 0.02, out_pad_size = 8, debug = False, filenames = None):
    '''
    Crop out the lung area from CXR\n
    lung prediction based on UNet: https://github.com/imlab-uiip/lung-segmentation-2d
    Parameters
    ----------
    data_path : preferrebly pathlib object
        all images in that path will be loaded for lung segmentation if filenames not specified.
    result_folder : preferrebly pathlib object
        path to output
    UNet: preferrebly pathlib object
        path to UNet
    debug_folder : preferrebly pathlib object
        path to debug images; if not specified, no debug images will be written to local
    k_fold: int
        Specify how many processes to create to finish this task. 
        If None, processes are created based on adjust_process_num function
    cut_thresh: float
        connected components less than cut_thresh * np.prod(im_shape) will be removed
    out_pad_size: int
        Default to be 8, how many pixels to enlarge the bounding box. 
    debug: bool
        Default to be false. If true, will plot debugging images to screen instead of saving to local.
    filenames: list
        If specified, load these images instead of loading all images in data_path. 
        Absolute paths needed.
    '''
    data_path = pathlib.Path(data_path)
    result_folder = pathlib.Path(result_folder)
    result_folder.mkdir(parents=True, exist_ok=True)
    
    if not debug_folder is None:
        debug_folder = pathlib.Path(debug_folder)
        debug_folder.mkdir(parents=True, exist_ok=True)

    im_shape = (256, 256)
    chunknum = 0
    chunksize = 500
    print('processing data in ' + str(data_path))
    if filenames is None:
        filenames = list(data_path.glob('*'))
    
    totalfiles = len(filenames)
    while chunknum * chunksize < totalfiles:
        start = chunknum * chunksize
        end =  min(totalfiles, (chunknum + 1) * chunksize)
        print('segmenting {} files of folder {}'.format((start, end), str(data_path)))
        curfiles = filenames[start : end]        

        if debug:
            lungseg_one_process(result_folder, UNet, curfiles, 
                            im_shape = im_shape, debug_folder = debug_folder, cut_thresh = cut_thresh, out_pad_size = out_pad_size, debug = debug)
            return

        start_time = time.time()
        keywords = {
            'im_shape' : im_shape,
            'cut_thresh' : cut_thresh,
            'out_pad_size' : out_pad_size,
            'debug_folder' : debug_folder,
            'debug' : debug
        }
        
        if k_fold is None:
            k_fold = adjust_process_num(len(curfiles))

        print('Running using {} process'.format(k_fold))
        start_idxs = gen_idx(len(curfiles), k_fold)
        pool = []
        for k in range(k_fold):
            # attention here the slicing is wrong!!
            # we missed the last a few images
            arg_str = (
                result_folder, 
                UNet, 
                curfiles[start_idxs[k]: start_idxs[k + 1]]
            )
            p = multiprocessing.Process(target = lungseg_one_process, args = arg_str, kwargs = keywords)
            p.start()
            pool.append(p)    

        for p in pool:
            p.join()
        print('{} processes takes {} seconds'.format(k_fold, time.time() - start_time))
        chunknum = chunknum + 1


def genlist(data_path_list, list_dict):
    cur_dict = list_dict
    for i in range(len(data_path_list)):
        if i > 0:
            cur_dict = cur_dict[data_path_list[i]]
            # print(cur_dict)
            if type(cur_dict) == list:
                return cur_dict

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help = 'the directory of the image folder')
    parser.add_argument('-U', '--Unet', type = str, help = 'the directory of the saved Unet weights')
    parser.add_argument('-o', '--output', type = str, help = 'the directory of the resized image')
    args = parser.parse_args()
    UNet_path = args.Unet
    folder_path = os.path.normpath(args.folder)
    output_path = os.path.normpath(args.output)



    if not os.path.isdir(folder_path):
        containing_folder = os.path.dirname(folder_path)
        singlefolder_lungseg(containing_folder, output_path, UNet_path, out_pad_size=8, debug=False, filenames=[pathlib.Path(folder_path)])
    else:
        singlefolder_lungseg(folder_path, output_path, UNet_path, out_pad_size=8, debug=False)
    print('Completed!')
    

    # # single image lung segmentation
    # from keras.models import load_model
    # parent_path = pathlib.Path(__file__).absolute().parent
    # data_path = parent_path.parent.joinpath('NMHFiles_sample', 'Negative')
    # img_path = data_path.joinpath('8356_47cfe01e37c2237dd6a31b424473c89f_AP_2.png')

    # UNet = load_model(UNet_path)
    # im_shape = (256, 256)
    # X, resized_raw, raw_images = load_CXR_from_list([img_path], im_shape)

    # result_folder = parent_path.parent.joinpath('NMHFiles_sample_crop', 'Negative')

    # single_img_crop(X[0], resized_raw[0], raw_images[0], img_path, UNet, result_folder, debugging = True)

    # print('Total time = {}'.format(time.time() - start))

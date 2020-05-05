import numpy as np
from skimage import morphology, io, color, exposure, img_as_float, transform, util
from matplotlib import pyplot as plt
import pathlib
import cv2
import time
import multiprocessing

def load_CXR_from_files(data_path, filenames, im_shape):
    filelist = []
    for f in filenames:
        filelist.append(data_path.joinpath(f))
    
    print('Loading from files : {}'.format(filelist))
    return load_CXR_from_list(filelist, im_shape), filelist

def load_CXR_from_folder(data_path, im_shape):
    filelist = list(data_path.glob('*'))
    print('Loading folder Dataset : {}'.format(str(data_path)))
    return load_CXR_from_list(filelist, im_shape), filelist

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
    color_mask[mask == 1] = [0, 0, 1]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

def get_parent_path(script):
    cur_path = pathlib.Path(script).absolute()
    parent_path = cur_path.parent.parent

    return parent_path

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

def select_lung(pred, resized_raw, cut_thresh, debug, out_pad_size, k_size = 5):
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
    opened = np.zeros(opened.shape, dtype = np.uint8)
    for i in range(1, min(stats.shape[0], 3)):
        opened[label_map == idx_sorted[i]] = 1

    if stats.shape[0] < 3:
        if stats.shape[0] == 1:
            return opened, None
        else:
            print('Single Lung Detected !!!')
            top = stats[1, cv2.CC_STAT_TOP]
            bottom = stats[1, cv2.CC_STAT_TOP] + stats[1, cv2.CC_STAT_HEIGHT]
            left = stats[1, cv2.CC_STAT_LEFT]
            right = stats[1, cv2.CC_STAT_LEFT] + stats[1, cv2.CC_STAT_WIDTH]

            spine_pos = select_spine(resized_raw)
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

    # expand the bounding box a little bit
    left = max(0, left - out_pad_size)
    right = min(opened.shape[1] - 1, right + out_pad_size)
    top = max(0, top - out_pad_size)
    bottom = min(opened.shape[0] - 1, bottom + out_pad_size)

    bbox = np.array([left, top, right, bottom])
    boxed = cv2.rectangle(opened, (left, top), (right, bottom), color = 1, thickness=3)
    
    return opened, bbox

def crop_lung(raw_img, cur_shape, bbox):
    if bbox is None:
        print('NO qualified lung area selected !! Original image outputted.')
        return raw_img
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

    return lung_img


def lungseg_fromdata(X, resized_raw, raw_images, file_paths, UNet, result_folder, 
                        im_shape = (256, 256), cut_thresh = 0.02, out_pad_size = 3, debug_folder = None ,debugging = False):
    from keras.models import load_model
    from keras.preprocessing.image import ImageDataGenerator
    import keras
    import tensorflow as tf

    n_test = X.shape[0]
    inp_shape = X[0].shape
    UNet = load_model(UNet)
    print('n_test = {}'.format(n_test))
    # For inference standard keras ImageGenerator can be used.
    test_gen = ImageDataGenerator(rescale=1.)
    no_lung_detected = []
    
    i = 0
    if not (debug_folder is None) or debugging:
        fig, axes = plt.subplots(2, 3, figsize = (12, 8))
    
    for xx in test_gen.flow(X, batch_size=1, shuffle=False):
        # print(xx.dtype, xx.shape)
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
        pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
        
        pr = (pred > 0.5).astype(np.uint8)
        denoised, bbox = select_lung(pr, resized_raw[i], cut_thresh = cut_thresh, debug = debugging, out_pad_size = out_pad_size)
        lung_img = crop_lung(raw_images[i], im_shape, bbox)
        
        if bbox is None:
            no_lung_detected.append(file_paths[i])
        
        filename = file_paths[i].stem
        suffix = file_paths[i].suffix
        print('output result for ' + filename)
        # print('img.shape = ', img.shape)
        if not (debug_folder is None) or debugging:
            axes[0, 0].set_title(filename + '\n'+ '_resized_raw')
            axes[0, 0].imshow(resized_raw[i], cmap='gray')
            axes[0, 0].set_axis_off()

            axes[0, 1].set_title(filename + '\n'+ '_processed')
            axes[0, 1].imshow(img, cmap='gray')
            axes[0, 1].set_axis_off()

            axes[0, 2].set_title(filename + '\n'+ '_rawpred')
            axes[0, 2].imshow(pr ,cmap='gray')
            axes[0, 2].set_axis_off()

            axes[1, 0].set_title(filename + '\n'+ '_denoised_pred')
            axes[1, 0].imshow(denoised, cmap='gray')
            axes[1, 0].set_axis_off()

            axes[1, 1].set_title(filename + '\n'+ '_raw_masked')
            axes[1, 1].imshow(masked(resized_raw[i], pr, alpha = 0.6))
            axes[1, 1].set_axis_off()
            
            axes[1, 2].set_title(filename + '\n'+ '_denoised_masked')
            axes[1, 2].imshow(masked(resized_raw[i], denoised, alpha = 0.6))
            axes[1, 2].set_axis_off()

            plt.tight_layout()
            if not (debug_folder is None):
                out_path = debug_folder.joinpath(filename + '_debug' + suffix)
        
            if not debugging:
                plt.savefig(str(out_path))
                out_path = debug_folder.joinpath(filename + '_crop' + suffix)
                io.imsave(str(out_path), lung_img)
            else:
                plt.figure()

        if not debugging:
            out_path = result_folder.joinpath(filename + '_crop' + suffix)
            io.imsave(str(out_path), lung_img )
        else:
            plt.imshow(lung_img, cmap = 'gray')
            plt.show()

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
    if length < 100:
        k_fold = 4
    elif length < 400:
        k_fold = 8
    elif length < 1000:
        k_fold = 16
    else:
        k_fold = 24

    return k_fold

def singlefolder_lungseg(data_path, result_folder, UNet, debug_folder = None, k_fold = None, cut_thresh = 0.02, out_pad_size = 3, debug = False, filenames = None):
    '''
    Crop out the lung area from CXR\n
    lung prediction based on UNet: https://github.com/imlab-uiip/lung-segmentation-2d


    Parameters
    ----------
    data_path : preferrebly pathlib object
        all images in that path will be loaded for lung segmentation. 
    result_folder : preferrebly pathlib object
        path to output
    UNet: preferrebly pathlib object
        path to UNet
    debug_folder : preferrebly pathlib object (default None)
        if given, will save debug images to local; default not ouputting.
    k_fold: int
        Specify how many processes to create to finish this task. 
        If None, processes are created based on adjust_process_num function
    cut_thresh: float
        connected components less than cut_thresh * im_shape[0] * im_shape[1] will be removed
    out_pad_size: int
        Default to be three, how many pixels to enlarge the bounding box. 
    debug: bool
        Default to be false. If true, will plot debugging images to screen instead of saving to local.
    filenames: list
        If specified, load these images instead of loading all images in data_path

    '''
    data_path = pathlib.Path(data_path)
    result_folder = pathlib.Path(result_folder)
    result_folder.mkdir(parents=True, exist_ok=True)
    
    if not debug_folder is None:
        debug_folder = pathlib.Path(debug_folder)
        debug_folder.mkdir(parents=True, exist_ok=True)

    im_shape = (256, 256)
    if filenames is None:
        [X, resized_raw, raw_images], file_paths = load_CXR_from_folder(data_path, im_shape)
    else:
        [X, resized_raw, raw_images], file_paths = load_CXR_from_files(data_path, filenames, im_shape)
    print('X shape = ', X.shape)  

    if debug:
        lungseg_fromdata(X, resized_raw, raw_images, file_paths, UNet, result_folder, 
                    im_shape = im_shape, cut_thresh = cut_thresh, out_pad_size = out_pad_size, debug_folder = None, debugging = debug)
        return

    start_time = time.time()
    keywords = {
        'im_shape' : im_shape,
        'cut_thresh' : cut_thresh,
        'out_pad_size' : out_pad_size,
        'debug_folder' : debug_folder,
        'debugging' : debug
    }

    if k_fold is None:
        k_fold = adjust_process_num(len(file_paths))

    print('Running using {} process'.format(k_fold))

    start_idxs = gen_idx(len(file_paths), k_fold)
    pool = []
    for k in range(k_fold):
        # attention here the slicing is wrong!!
        # we missed the last a few images
        p = multiprocessing.Process(target = lungseg_fromdata, args = ( X[start_idxs[k] : start_idxs[k+1] , ...],
                                                                        resized_raw[start_idxs[k] : start_idxs[k+1] , ...],
                                                                        raw_images[start_idxs[k] : start_idxs[k+1] ],
                                                                        file_paths[start_idxs[k] : start_idxs[k+1] ],
                                                                        UNet, 
                                                                        result_folder),
                                                                kwargs = keywords)
        p.start()
        pool.append(p)    

    for p in pool:
        p.join()
    print('{} processes takes {} seconds'.format(k_fold, time.time() - start_time))


def genlist(data_path_list, list_dict):
    cur_dict = list_dict
    for i in range(len(data_path_list)):
        if i > 0:
            cur_dict = cur_dict[data_path_list[i]]
            # print(cur_dict)
            if type(cur_dict) == list:
                return cur_dict

if __name__ == '__main__':
    
    # UNet = load_model( str(pathlib.Path(__file__).parent.joinpath('trained_model.hdf5')) )
    # # UNet._make_predict_function()
    # xx = np.zeros((1, 256, 256, 1), dtype=np.float32)
    # UNet.predict(xx)

    # UNet._make_predict_function()
    # UNet = CNN('trained_model.hdf5')
    UNet_path = str(pathlib.Path(__file__).parent.joinpath('trained_model.hdf5'))
    level1folders = ['Test', 'Training', 'Validation']
    level2folders = ['Pos', 'Neg']

    data_path_list = ['New_Data', 'Validation', 'Pos']
    attached_str = '_crop'
    parent_path = pathlib.Path(__file__).absolute().parent
    data_path = join_path_from_list(parent_path, data_path_list)
    debug_folder = join_path_from_list(parent_path, change_first_folder(data_path_list, attached_str = attached_str + '_debug'))    
    result_folder = join_path_from_list(parent_path, change_first_folder(data_path_list, attached_str = attached_str))
    # for f1 in level1folders:
    #     for f2 in level2folders:
    #         data_path_list = ['New_Data', f1, f2]
    #         singlefolder_lungseg_parallel(data_path_list, UNet_path, attached_str = '_cut0.015', cut_thresh = 0.015)
    singlefolder_lungseg(data_path, result_folder, UNet_path)

    # dp = ['New_Data', 'Test', 'Neg'11111]
    # singlefolder_lungseg_parallel(dp, UNet_path, attached_str = '_cut0.15', debug = True, filenames=['person675_bacteria_2569.jpeg'])

    # singlefolder_lungseg(['New_Data', 'Validation', 'Neg'], UNet)
    # singlefolder_lungseg(['New_Data', 'Training', 'Neg'], UNet)   

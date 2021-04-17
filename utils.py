import copy
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image
from tifffile import tifffile
from skimage.segmentation import slic



def openImage(img_path):
    if "jpg" in img_path or "png" in img_path:
        image = Image.open(img_path).convert('RGB')  # return Image object
    elif "tiff" in img_path or "tif" in img_path:
        image = tifffile.imread(img_path)  # returns numpy array
    else:
        raise TypeError("The input image format doesn\'t support, we only support png, jpg and tif/tiff format ")

    return image

def SP_fusion(image1, image2, n_segments, compactness, merge, merge_regions=50, img_names=('', '')):
    """
    :param image1: image of time1
    :param image2: image of time2
    :param n_segments:
    :param compactness:
    :param merge:
    :param merge_regions
    :param img_names
    :return: the fused superpixel of image1 and image2
    """
    # SLIC Superpixel and save
    labels1 = slic(np.array(image1), n_segments, compactness)
    labels2 = slic(np.array(image2), n_segments, compactness)

    # result_nomerge = [labels1, labels2]
    if merge:
        import skimage.external.tifffile as tifffile
        from os.path import abspath, dirname
        Maindirc = abspath(dirname(__file__))

        # set constant for superpixels merge
        sys.path.insert(0, 'MergeTool')

        MergeTool = 'SuperPixelMerge.exe'
        SP_label = Maindirc + '/Superpixel.tif'
        SPMG_label = Maindirc + '/Merge.tif'
        MG_Criterion = 3  # 0, 1, 2, 3
        Num_of_Region = merge_regions  # the number of regions after region merging
        MG_Shape = 0.7
        MG_Compact = 0.7

        result = []
        for img_name, labels in zip(img_names, [labels1, labels2]):
            # SLIC Superpixel and save
            labels = labels.astype('int32')
            tifffile.imsave('Superpixel.tif', labels, photometric='minisblack')

            # Call the Superpixel Merge tool, format the command line input
            os.chdir('/data/Project_prep/superpixel-cosegmentation/MergeTool/')
            cmd_line = '{} {} {} {} {} {} {} {} {}'.format(MergeTool, img_name, SP_label, SPMG_label,
                                                           MG_Criterion, Num_of_Region, MG_Shape, ' ',
                                                           MG_Compact)
            # print('cmd_line', cmd_line)
            os.system(cmd_line)  # call the Superpixel Merge Tool
            os.chdir('..')

            # save merged slic labels
            MG_labels = tifffile.imread(SPMG_label)
            result.append(MG_labels)

        labels1, labels2 = result

    fusion_labels_after = labels1 + labels2 * 100

    return fusion_labels_after, labels1, labels2

def sp_accuracy(sp, label):
    """
    :param sp:
    :param label:
    :return: the accuracy of superpixel based on the label
    """
    if len(label.shape) == 3:
        label = RGB2Index(label)
    sp_pred = classOfSP(sp, label)
    correct = sp_pred[sp_pred == label]
    acc = len(correct) / sp_pred.size

    return round(acc, 3)


def classOfSP(sp, prediction):
    """
    :param sp: super pixel label of a image | type: <numpy.ndarray>
    :param prediction: the probability of segmented result | type: list 200*200
    :return: the segmented result as super pixel | type: list
    """
    outset = np.unique(sp.flatten())  # the unique labels
    fuse_prediction = copy.deepcopy(prediction)
    for i in outset:
        mostpred, times = Counter(prediction[sp == i]).most_common(1)[0]
        fuse_prediction[sp == i] = mostpred

    return fuse_prediction


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def RGB2Index(label, mode='palatte'):
    """
    :param label: the ndarray with RGB value needs to be transfer into array with index value
    :param mode: whether turn based on palatte or only find Non-zeros
    :return: the numpy int labels
    """
    if len(label.shape) == 2:
        return label
    palette = list([[0, 0, 0], [150, 250, 0], [0, 250, 0], [0, 100, 0],
                    [200, 0, 0], [255, 255, 255], [0, 0, 200], [0, 150, 250]])
    h, w, c = label.shape
    label = label.tolist()
    label_int = copy.deepcopy(label)
    for i in range(h):
        for j in range(w):
            if 'palatte' == mode:
                try:
                    idx = palette.index(label[i][j])
                except ValueError:
                    print('the value {} is not in palatte', label[i][j])
                    idx = 255
            elif mode == 'NonZero':
                idx = [0, 1][label[i][j] != [0, 0, 0]]
            label_int[i][j] = idx

    return np.array(label_int)

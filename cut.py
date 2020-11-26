"""
cut all pixels into 800*800 and give a good name
"""
import copy
from glob import glob

from PIL import Image
from pathlib import Path
import numpy as np
import os

palette_save = [0, 0, 0,
                0, 0, 0,
                150, 250, 0,
                0, 250, 0,
                0, 100, 0,
                200, 0, 0,
                255, 255, 255,
                0, 0, 200,
                0, 150, 250]

mappalette = {(0, 1, 0): 0, (0, 0, 0): 0, (0, 0, 5): 0,
              (191, 255, 0): 1, (192, 255, 0): 1, (150, 250, 0): 1,
              (0, 255, 0): 2, (0, 250, 0): 2,
              (0, 102, 0): 3, (0, 86, 0): 1, (0, 153, 0): 1, (0, 100, 0): 1,
              (255, 0, 0): 4, (200, 0, 0): 4,
              (255, 255, 255): 5, (255, 255, 0): 5,
              (0, 0, 204): 6, (0, 0, 255): 6, (0, 0, 200): 6, (0, 2, 255): 6,
              (0, 86, 255): 7, (0, 153, 255): 7, (0, 150, 250): 7}

palette = list([[0, 0, 0], [150, 250, 0], [0, 250, 0], [0, 100, 0],
                [200, 0, 0], [255, 255, 255], [0, 0, 200], [0, 150, 250]])


# def test():
#     lbl_ext = "png"
#     label_files = sorted(glob(os.path.join('label', f'*.{lbl_ext}')))
#     label1 = np.array(Image.open(label_files[0]).convert('RGB'))
#     label2 = np.array(Image.open(label_files[1]).convert('RGB'))
#     change_gt = np.zeros(label1.shape)
#     label1, label2 = label2intarray(label1, trans_palette=mappalette), \
#                      label2intarray(label2, trans_palette=mappalette)
#     change_gt[label1 != label2] = 1
#     change_gt[label1 == 0] = 0
#     change_gt[label2 == 0] = 0
#     save_images(change_gt, cat_dir, labellist[0].stem, 'change_gt')


# def change_gt(index, save_dir):
#     # find gt
#     label1 = np.array(Image.open(labellist[index]).convert('RGB'))
#     label2 = np.array(Image.open(labellist[index + 1]).convert('RGB'))
#
#     label1, label2 = label2intarray(label1, trans_palette=mappalette), \
#                      label2intarray(label2, trans_palette=mappalette)
#     change_gt = np.zeros(label1.shape)
#     change_gt[label1 != label2] = 1
#     change_gt[label1 == 0] = 0
#     change_gt[label2 == 0] = 0
#     save_images(change_gt, save_dir, labellist[index].stem, 'change_gt')


def label2intarray(label, trans_palette):
    """

    :param trans_palette:
    :param label: the ndarray needs to be transfer into int-element-array
    :return:
    """
    if len(label.shape) == 2:
        return label

    h, w, c = label.shape
    label = label.tolist()
    label_int = copy.deepcopy(label)
    error = set()
    for i in range(h):
        for j in range(w):
            try:
                idx = trans_palette[tuple(label[i][j])]
                label_int[i][j] = idx
            except KeyError:
                error.add(tuple(label[i][j]))
    print('error', error)
    return np.array(label_int)


def colorize_mask(mask, palette):
    """
    mask should only have 2 channels to get the right result
    """
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)  # palette 中每三个代表像素值为1，2，3...
    return new_mask


def save_images(mask, output_path=None, image_file=None, tag=None, savePath=None):
    colorized_mask = colorize_mask(mask, palette_save)
    if savePath is None:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        image_file = os.path.basename(image_file).split('.')[0]
        savePath = os.path.join(output_path, image_file + tag + '.png')
    colorized_mask.save(savePath)


def cutsave(name, im, imgtag=False):
    save_name = 'Hid{}_Wid{}_'.format(i, j) + name
    if not imgtag:
        crop = im[i * height:(i + 1) * height, j * width:(j + 1) * width, :1]
        crop = np.squeeze(crop, axis=2)
        crop = colorize_mask(crop, palette_save)
    else:
        crop = im[i * height:(i + 1) * height, j * width:(j + 1) * width]
        crop = Image.fromarray(crop)
    crop.save(os.path.join(sp, save_name))
    print('save to', os.path.join(sp, save_name))


read_path = Path(r'E:\School\change detection\co-segmentation\OriginData\shijiazhuang')
save_path = Path(r'E:\School\change detection\co-segmentation\Data\shijiazhuang')
height, width = 800, 800

order_first = ['GF2_PMS1_E114.6_N38.0_20170906_L1A0002588959-MSS1_1',
               'GF2_PMS1_E114.6_N38.0_20170906_L1A0002588959-MSS1_2',
               'GF2_PMS1_E114.6_N38.0_20170906_L1A0002588959-MSS1_3',
               'GF2_PMS1_E114.6_N38.0_20180416_L1A0003126155-MSS1_1',
               'GF2_PMS1_E114.6_N38.0_20180416_L1A0003126155-MSS1_2']

order_second = ['GF2_PMS2_E114.7_N37.9_20151002_L1A0001074088-MSS1_1',
                'GF2_PMS2_E114.7_N37.9_20151002_L1A0001074088-MSS1_2',
                'GF2_PMS2_E114.7_N37.9_20151002_L1A0001074088-MSS1_3',
                'GF2_PMS2_E114.7_N37.9_20151002_L1A0001074088-MSS1_4',
                'GF2_PMS2_E114.7_N37.9_20151002_L1A0001074088-MSS1_5']

from tifffile import tifffile

cnt = 1
for index in range(len(order_second)):
    # cut and save
    img1_path = os.path.join(read_path, 'first\\images\\' + order_first[index] + '.png')
    img2_path = os.path.join(read_path, 'second\\images\\' + order_second[index] + '.png')
    img1, img2 = np.array(Image.open(img1_path)), np.array(Image.open(img2_path))
    print('image_name', img1_path, img2_path, img1.shape)
    im_height, im_width, channel = img1.shape

    lbl1_path = os.path.join(read_path, 'first\\labels_gray\\' + order_first[index] + '.png')
    lbl2_path = os.path.join(read_path, 'second\\labels_gray\\' + order_second[index] + '.png')
    lbl1, lbl2 = np.array(Image.open(lbl1_path)), np.array(Image.open(lbl2_path))

    for i in range(im_height // height):
        for j in range(im_width // width):
            # save directory
            sp = save_path.joinpath('{}'.format(cnt))
            if not sp.exists():
                sp.mkdir()
            cutsave(order_first[index] + '.jpg', img1, imgtag=True)
            cutsave(order_second[index] + '.jpg', img2, imgtag=True)
            cutsave(order_first[index] + '.png', lbl1)
            cutsave(order_second[index] + '.png', lbl2)
            cnt += 1
    # break

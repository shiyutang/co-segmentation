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


def test():
    lbl_ext = "png"
    label_files = sorted(glob(os.path.join('label', f'*.{lbl_ext}')))
    label1 = np.array(Image.open(label_files[0]).convert('RGB'))
    label2 = np.array(Image.open(label_files[1]).convert('RGB'))
    change_gt = np.zeros(label1.shape)
    label1, label2 = label2intarray(label1, trans_palette=mappalette), \
                     label2intarray(label2, trans_palette=mappalette)
    change_gt[label1 != label2] = 1
    change_gt[label1 == 0] = 0
    change_gt[label2 == 0] = 0
    save_images(change_gt, cat_dir, labellist[0].stem, 'change_gt')


def change_gt(index, save_dir):
    # find gt
    label1 = np.array(Image.open(labellist[index]).convert('RGB'))
    label2 = np.array(Image.open(labellist[index + 1]).convert('RGB'))

    label1, label2 = label2intarray(label1, trans_palette=mappalette), \
                     label2intarray(label2, trans_palette=mappalette)
    change_gt = np.zeros(label1.shape)
    change_gt[label1 != label2] = 1
    change_gt[label1 == 0] = 0
    change_gt[label2 == 0] = 0
    save_images(change_gt, save_dir, labellist[index].stem, 'change_gt')


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
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def save_images(mask, output_path=None, image_file=None, tag=None, savePath=None):
    colorized_mask = colorize_mask(mask, palette_save)
    if savePath is None:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        image_file = os.path.basename(image_file).split('.')[0]
        savePath = os.path.join(output_path, image_file + tag + '.png')
    colorized_mask.save(savePath)


read_path = Path(r'D:\school\change detection\co-segmentation\xionganData')
dir_list = [read_path.joinpath(f) for f in read_path.glob('0*')]
print('dir_list', dir_list)

height, width = 800, 800

for cat_dir in dir_list:
    cat_dir = Path(cat_dir)
    if '007' not in str(cat_dir):
        continue

    image_list = [f for f in cat_dir.glob('*[^segimg]')]
    print(len(image_list))

    savedir = cat_dir.joinpath('segimg')
    if not savedir.exists():
        savedir.mkdir()

    # for image_name in image_list:
    #     print('image_name', image_name)
    #     im = Image.open(image_name)
    #     im = np.array(im)
    #     im_height, im_width, channel = im.shape
    #     im_name = str(image_name).split('\\')[-1]
    #     if 'label' in str(image_name):
    #         save_ext = '.png'
    #     else:
    #         save_ext = '.jpg'
    #
    #     for i in range(im_height // height):
    #         for j in range(im_width // width):
    #             save_name = 'Hid{}_Wid{}_'.format(i, j) + im_name[:-4] + save_ext
    #             print('start save {}'.format(save_name))
    #             crop = im[i * height:(i + 1) * height, j * width:(j + 1) * width, :]
    #             crop_im = Image.fromarray(crop)
    #             crop_im.save(os.path.join(savedir, save_name))
    labellist = [f for f in savedir.glob('*label*')]
    print('labellist', labellist)

    # test()
    # for i in range(len(labellist) // 2):
    #     change_gt(index=2 * i, save_dir=savedir)

    for labelname in labellist:
        label = np.array(Image.open(labelname).convert('RGB'))
        label = label2intarray(label, trans_palette=mappalette)
        save_images(label, savePath=labelname)

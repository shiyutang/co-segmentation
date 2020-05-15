"""
cut all pixels into 800*800 and give a good name
"""
import copy
from glob import glob

from PIL import Image
from pathlib import Path
import numpy as np
import os

palette = [0, 0, 0,
           150, 250, 0,
           0, 250, 0,
           0, 100, 0,
           200, 0, 0,
           255, 255, 255,
           0, 0, 200,
           0, 150, 250]

mappalette = [
    [9, 254, 1],  # 0 250 0
    [254, 0, 0],  # 200 0 0
    [255, 249, 251],  # 255 255 255
]  #todo finish map


def label2intarray(label):
    """

    :param label: the ndarray needs to be transfer into int-element-array
    :return:
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
            idx = palette.index(label[i][j])
            label_int[i][j] = idx

    return np.array(label_int)


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def save_images(mask, output_path, image_file, tag):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    image_file = os.path.basename(image_file).split('.')[0]
    colorized_mask = colorize_mask(mask, palette)
    save_path = os.path.join(output_path, image_file + tag + '.png')
    colorized_mask.save(save_path)


read_path = Path(r'D:\school\change detection\co-segmentation\xionganData')
dir_list = [read_path.joinpath(f) for f in read_path.glob('0*')]

print('dir_list', dir_list)

height, width = 800, 800
save_ext = '.jpg'


def test():
    lbl_ext = "png"
    label_files = sorted(glob(os.path.join('label', f'*.{lbl_ext}')))
    label1 = np.array(Image.open(label_files[0]).convert('RGB'))
    label2 = np.array(Image.open(label_files[1]).convert('RGB'))
    change_gt = np.zeros(label1.shape)
    label1, label2 = label2intarray(label1), label2intarray(label2)
    change_gt[label1 != label2] = 1
    change_gt[label1 == 0] = 0
    change_gt[label2 == 0] = 0
    save_images(change_gt, cat_dir, labellist[0].stem, 'change_gt')


def change_gt(index, save_dir):
    # find gt
    label1 = np.array(Image.open(labellist[index]).convert('RGB'))
    label2 = np.array(Image.open(labellist[index+1]).convert('RGB'))
    label1, label2 = label2intarray(label1), label2intarray(label2)
    change_gt = np.zeros(label1.shape)
    change_gt[label1 != label2] = 0
    change_gt[label1 == 0] = 0
    change_gt[label2 == 0] = 0
    save_images(change_gt, save_dir, labellist[index].stem, 'change_gt')


for cat_dir in dir_list:
    cat_dir = Path(cat_dir)

    image_list = [f for f in cat_dir.glob('*[^segimg]')]
    print(len(image_list))

    savedir = cat_dir.joinpath('segimg')
    if not savedir.exists():
        savedir.mkdir()

    for image_name in image_list:
        im = Image.open(image_name)
        # im.show()
        im = np.array(im)
        print('image_name', image_name)
        im_height, im_width, channel = im.shape
        im_name = str(image_name).split('\\')[-1]
        for i in range(im_height // height):
            for j in range(im_width // width):
                save_name = 'Hid{}_Wid{}_'.format(i, j) + im_name[:-4] + save_ext
                print('start save {}'.format(save_name))
                crop = im[i * height:(i + 1) * height, j * width:(j + 1) * width, :]
                crop_im = Image.fromarray(crop)
                crop_im.save(os.path.join(savedir, save_name))
    labellist = [f for f in savedir.glob('*label*')]
    print('labellist', labellist)

    # test()
    for i in range(len(labellist)//2):
        change_gt(index=2*i, save_dir=savedir)

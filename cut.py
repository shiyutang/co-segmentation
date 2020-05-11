"""
cut all pixels into 800*800 and give a good name
"""

from PIL import Image
from pathlib import Path
import numpy as np
import os

# read_path = 'D:\school\change detection\data\change_chengdu_part\mask'
from skimage.external.tifffile import tifffile

read_path = r'D:\school\change detection\superpixel-cosegmentatin\xionganData'
image_list = [f for f in Path(read_path).glob('*.tif')]
print('image_list', image_list)

height, width = 800, 800
save_ext = '.jpg'

for image_name in image_list:
    # im = Image.open(image_name)
    # im.show()
    im = tifffile.imread(image_name)
    im = np.array(im)
    im_height, im_width, channel = im.shape
    im_name = str(image_name).split('\\')[-1]
    for i in range(im_height//height):
        for j in range(im_width//width):
            save_name = 'Hid{}_Wid{}_'.format(i,j) + im_name[:-4]+save_ext
            print('start save {}'.format(save_name))
            crop = im[i*height:(i+1)*height, j*width:(j+1)*width, :]
            crop_im = Image.fromarray(crop)
            crop_im.save(os.path.join(read_path, save_name))

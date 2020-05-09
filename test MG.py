from skimage.segmentation import slic, mark_boundaries
import skimage.external.tifffile as tifffile
import os
import sys
from os.path import abspath, dirname
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt

Maindirc = abspath(dirname(__file__))
print('maindirc',Maindirc)
# 原图路径
picPath = 'test'
# 超像素保存路径
savePath = 'sp'

# set constant for superpixels merge
sys.path.insert(0, 'MergeTool')
MergeTool = 'SuperPixelMerge.exe'
SP_label = Maindirc+'/Superpixel.tif'
SPMG_label = Maindirc+'/Merge.tif'
MG_Criterion = 3  # 0, 1, 2, 3
Num_of_Region = 30 # the number of regions after region merging
MG_Shape = 0.7
MG_Compact = 0.7


pics = [picPath + '/' + i for i in os.listdir(picPath)]
for i, img_name in enumerate(pics):
    # img = io.imread(img_name)
    img = Image.open(img_name).convert('RGB')

    ## SLIC Superpixel and save
    labels = slic(img, n_segments=500, compactness=10)
    labels = labels.astype('int32')
    out = mark_boundaries(img, labels)
    io.imshow(out)
    plt.show()
    tifffile.imsave('Superpixel.tif', labels, photometric='minisblack')


    ## Call the Superpixel Merge tool, format the command line input
    os.chdir('./MergeTool/')
    cmd_line = '{} {} {} {} {} {} {} {} {}'.format(MergeTool,img_name,SP_label,SPMG_label,\
                                                   MG_Criterion,Num_of_Region,MG_Shape,' ',MG_Compact)
    os.system(cmd_line)  # call the Superpixel Merge Tool
    os.chdir('..')

    ## save merged slic labels
    MG_labels = tifffile.imread(SPMG_label)
    out = mark_boundaries(img, MG_labels)
    io.imshow(out)
    plt.show()

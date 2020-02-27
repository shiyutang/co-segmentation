from skimage.segmentation import slic, find_boundaries,mark_boundaries
import skimage.external.tifffile as tifffile
import pickle
import os
import sys
from os.path import abspath,dirname
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

def SP_merge(image):
    ## SLIC Superpixel and save
    img = np.array(image)
    labels = slic(img, n_segments=500, compactness=10)
    return labels


if __name__ == '__main__':
    #原图路径
    picPath = 'test'
    #超像素保存路径
    savePath = 'sp'

    Maindirc= abspath(dirname(__file__))
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

    for i,img_name in enumerate(pics):
        img = io.imread(img_name)

        ## SLIC Superpixel and save
        labels = slic(img, n_segments=500, compactness=10)
        labels = labels.astype('int32')
        out = mark_boundaries(img, labels)
        tifffile.imsave('Superpixel.tif', labels, photometric='minisblack')


        ## Call the Superpixel Merge tool, format the command line input
        os.chdir('./MergeTool/')
        cmd_line = '{} {} {} {} {} {} {} {} {}'.\
            format(MergeTool,img_name,SP_label,SPMG_label,\
                MG_Criterion,Num_of_Region,MG_Shape,' ',MG_Compact)
        os.system(cmd_line)  # call the Superpixel Merge Tool
        os.chdir('..')

        ## save merged slic labels
        MG_labels = tifffile.imread(SPMG_label)
        out = mark_boundaries(img, MG_labels)
        io.imshow(out)
        plt.show()


        print(MG_labels)

        # with open('{}/MGslicLabels_{}.pkl'.format(savePath, i), 'wb') as file:
        #     pickle.dump(MG_labels, file)
        # print('saved file {}/MGslicLabels_{}.pkl'.format(savePath, i))
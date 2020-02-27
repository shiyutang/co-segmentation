import numpy as np
import pickle
from skimage import io
import matplotlib.pyplot as plt
from skimage.segmentation import slic, find_boundaries,mark_boundaries
import os

sp_Path = 'sp'
savePath = 'sp'
sp_1 = open(os.path.join(sp_Path,'MGslicLabels_0.pkl'),'rb')
sp_1 = pickle.load(sp_1)
sp_1 = np.array(sp_1)

sp_2 = open(os.path.join(sp_Path,'MGslicLabels_1.pkl'),'rb')
sp_2 = pickle.load(sp_2)
sp_2 = np.array(sp_2)

sp = sp_1 + sp_2

with open(os.path.join(savePath,'MGslicLabels_3.pkl'), 'wb') as file:
    pickle.dump(sp, file)
print('saved file {}/MGslicLabels_3.pkl'.format(savePath))
img = io.imread(os.path.join('test','1.jpg'))
out = mark_boundaries(img, sp)

## test
print('out',out.max(),out.min(),sp)
outset = np.unique(sp.flatten())
print('set',outset)

io.imshow(out)
plt.show()
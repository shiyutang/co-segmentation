
from skimage import io
import pickle
import skimage.external.tifffile as tifffile
import numpy as np

# 超像素标签路径
sp_label_Path = r'C:\Users\96251\Desktop\superpixel\label'
sp_label_name = '1'

# 超像素路径
sp_Path = r'C:\Users\96251\Desktop\superpixel\sp'
sp_name = 'MGslicLabels_3'

# load超像素
sp = open(sp_Path + '/' + sp_name + '.pkl', 'rb')
sp = pickle.load(sp)
sp = np.array(sp)

# load超像素标签
sp_label = io.imread(sp_label_Path + '/' + sp_label_name + '.png')
sp_label = np.array(sp_label)


# 生成单个超像素的标签数组 → 组成list
sp_label_list = []
for i in range(0, sp.max(), 1):
    sp_label_temp = sp_label[sp == i]
    sp_label_list.append(sp_label_temp)

sp_pixel_num = []  # 每个超像素中含有像素数
sp_class = [] #每个超像素的类别

for j in range(0, sp.max(), 1):
    temp = sp_label_list[j].tolist()
    pixel_num = len(temp)
    sp_pixel_num.append(pixel_num)

    # void
    num0 = temp.count([0, 0, 0, 255])
    # field
    num1 = temp.count([150, 250, 0,255])
    # forest
    num2 = temp.count([0, 100, 0,255])
    # grass
    num3 = temp.count([0, 250, 0,255])
    # building
    num4 = temp.count([200, 0, 0,255])
    # road
    num5 = temp.count([255, 255, 255,255])
    # waterbody
    num6 = temp.count([0, 0, 200,255])
    # bareland
    num7 = temp.count([0,150,250,255])
    sp_classes = [num0, num1, num2, num3, num4, num5, num6, num7]
    sp_class.append(sp_classes.index(max(sp_classes)))


print('sp_class',sp_class)
print('sp_pixel_num',sp_pixel_num)

import gdal, os
import numpy as np
import cv2


def tif_weishen_transform1(img_dir):
    print(img_dir)
    data_gdal = gdal.Open(img_dir)

    im_width = data_gdal.RasterXSize
    im_height = data_gdal.RasterYSize
    im_bands = data_gdal.RasterCount
    im_data = data_gdal.ReadAsArray(0, 0, im_width, im_height)
    tmp_pic = np.zeros([im_height, im_width, im_bands])

    tmp_pic[:, :, 0] = im_data[0, 0:im_height, 0:im_width]
    tmp_pic[:, :, 1] = im_data[1, 0:im_height, 0:im_width]
    tmp_pic[:, :, 2] = im_data[2, 0:im_height, 0:im_width]
    tmp_pic[:, :, 3] = im_data[3, 0:im_height, 0:im_width]

    channel_0 = tmp_pic[:, :, 0]
    channel_1 = tmp_pic[:, :, 1]
    channel_2 = tmp_pic[:, :, 2]
    channel_3 = tmp_pic[:, :, 3]
    final_out = tmp_pic.copy()

    to_sort = np.array(channel_0.reshape((im_height * im_width, 1)))

    sorted_data = np.sort(to_sort, axis=0)
    all_pix_num = im_height * im_width
    up_percent = 2
    dowm_percent = 2
    up_yuzhi = sorted_data[int(all_pix_num * (1 - up_percent / 100))]

    down_yuzhi = sorted_data[int(all_pix_num * (dowm_percent / 100))]
    print(up_yuzhi, down_yuzhi)

    output_img = np.where(channel_0 >= up_yuzhi, up_yuzhi, channel_0)
    output_img = np.where(channel_0 <= down_yuzhi, 0, output_img)

    output_img = (output_img - down_yuzhi) / (up_yuzhi - down_yuzhi) * 255
    final_out[:, :, 0] = output_img

    to_sort = np.array(channel_1.reshape((im_height * im_width, 1)))

    sorted_data = np.sort(to_sort, axis=0)
    all_pix_num = im_height * im_width

    up_yuzhi = sorted_data[int(all_pix_num * (1 - up_percent / 100))]

    down_yuzhi = sorted_data[int(all_pix_num * (dowm_percent / 100))]

    print(up_yuzhi, down_yuzhi)
    output_img = np.where(channel_1 >= up_yuzhi, up_yuzhi, channel_1)
    output_img = np.where(output_img <= down_yuzhi, 0, output_img)

    output_img = (output_img - down_yuzhi) / (up_yuzhi - down_yuzhi) * 255
    final_out[:, :, 1] = output_img

    to_sort = np.array(channel_2.reshape((im_height * im_width, 1)))

    sorted_data = np.sort(to_sort, axis=0)
    all_pix_num = im_height * im_width

    up_yuzhi = sorted_data[int(all_pix_num * (1 - up_percent / 100))]

    down_yuzhi = sorted_data[int(all_pix_num * (dowm_percent / 100))]

    print(up_yuzhi, down_yuzhi)
    output_img = np.where(channel_2 >= up_yuzhi, up_yuzhi, channel_2)
    output_img = np.where(output_img <= down_yuzhi, 0, output_img)

    output_img = (output_img - down_yuzhi) / (up_yuzhi - down_yuzhi) * 255
    final_out[:, :, 2] = output_img

    to_sort = np.array(channel_3.reshape((im_height * im_width, 1)))

    sorted_data = np.sort(to_sort, axis=0)
    all_pix_num = im_height * im_width

    up_yuzhi = sorted_data[int(all_pix_num * (1 - up_percent / 100))]

    down_yuzhi = sorted_data[int(all_pix_num * (dowm_percent / 100))]
    print(up_yuzhi, down_yuzhi)
    output_img = np.where(channel_3 >= up_yuzhi, up_yuzhi, channel_3)
    output_img = np.where(output_img <= down_yuzhi, 0, output_img)

    output_img = (output_img - down_yuzhi) / (up_yuzhi - down_yuzhi) * 255
    final_out[:, :, 3] = output_img
    # final_out=np.uint8(final_out)
    # cv2.imencode('.jpg', final_out)[1].tofile(img_dir.replace('.tif', '.jpg'))
    # cv2.imwrite(img_dir.replace('.tif', '.jpg'), np.uint8(final_out[:, :, 0:3]))
    cv2.imwrite(img_dir.replace('.tif', '.jpg'), final_out[:, :, 0:3])

# def weishen():
dirlist = os.listdir(r'D:\data\XA_data\4ChannelTif')
for i in dirlist:
    print(i)
    tif_weishen_transform1(r'D:\data\XA_data\4ChannelTif\\' + i)

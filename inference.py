import argparse
import json
import os
import numpy as np
from PIL import Image
from glob import glob
from scipy import ndimage
from pathlib import Path
from tifffile import tifffile

import torch
import torch.nn as nn
import torch.nn.functional as function
from torchvision import transforms

from models import PSPNet
from utils import openImage, RGB2Index, classOfSP, colorize_mask, sp_accuracy, SP_fusion


# noinspection PyPep8Naming,PyAttributeOutsideInit
class ChangeDetection:
    def __init__(self, cfg):
        ################
        # EXP Settings #
        ################
        self.exp = cfg["Exp"]["name"]
        self.config = cfg
        self.record_spAcc = cfg['Output']["record_SP_Acc"]  # save superpixel accuracy to a txt file
        # SP merge params
        self.n_segments = cfg["SP_setting"]["n_segments"]
        self.compactness = cfg["SP_setting"]["compactness"]
        self.merge_regions = cfg["SP_setting"]["merge_regions"]
        self.merge = cfg["SP_setting"]["if_merge"]
        if self.merge:
            self.n_segments, self.merge_regions = [1000, 1500, 2000, 3000], [100, 150, 200]
        # SP 整形参数
        self.threshold = cfg["SP_setting"]["threshold"]
        self.ignore_pixels = cfg["SP_setting"]["ignore_pixels"]

        # Device
        self.use_gpu = cfg["Exp"]["use_gpu"]
        if torch.cuda.is_available() and self.use_gpu:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # output Data
        self.out_path = cfg['Output']["img_outpath"]
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        # input data
        self.datapath = cfg["Data"]["path"]
        self.num_pairs = len(glob(os.path.join(self.datapath, '*')))
        self.pairs_dir = sorted(glob(os.path.join(self.datapath, '*')))

        # Transform
        self.scales = cfg["Transform"]["scales"]
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(cfg["Transform"]["normalizeX"],
                                              cfg["Transform"]["normalizeY"])
        self.num_classes = cfg["Exp"]["classes"]
        self.palette = cfg["Exp"]["palette"]

        # Model
        self.model = PSPNet(self.config)
        checkpoint = torch.load(cfg["Exp"]["checkpoint"], map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        if 'module' in list(checkpoint.keys())[0] and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)
        # criterion
        self.criterion = torch.nn.CrossEntropyLoss()

        # lr scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.9, last_epoch=-1, verbose=True)

        # result
        self.omit_seg_SP, self.falseAlarm_seg_SP, self.pixAcc_seg_SP, self.f1_score_seg_SP = [], [], [], []
        self.omit_seg, self.falseAlarm_seg, self.pixAcc_seg, self.f1_score_seg = [], [], [], []
        self.omit_SP, self.falseAlarm_SP, self.pixAcc_SP, self.f1_score_SP = [], [], [], []

    def load_datapath_change_gt(self, index):
        """
        desc: 根据数据格式，载入数据。成都标签为 png 格式，其他为 tif 格式，
        index: 索引位置

        """
        self.image_paths = sorted(glob(os.path.join(self.pairs_dir[index], '*.jpg')))
        if 'chengdu' in self.datapath or 'shijiazhuang' in self.datapath:
            # 变化需要根据标签推断， 标签为 png
            label1_path, label2_path = sorted(glob(os.path.join(self.pairs_dir[index], '*.png')))
            label1 = np.array(Image.open(label1_path).convert('RGB'))
            label2 = np.array(Image.open(label2_path).convert('RGB'))
            label1, label2 = RGB2Index(label1), RGB2Index(label2)
            change_gt = np.zeros(label1.shape)
            change_gt[label1 != label2] = 1
            change_gt[label1 == 0] = 0  # ignore background
            change_gt[label2 == 0] = 0
            self.change_gt = change_gt

        elif 'xiongan' in self.datapath:
            # 变化直接有标注的真值
            mask_path = glob(os.path.join(self.pairs_dir[index], 'mask/*.tif'))[0]  # 只取第一张
            change_gt = tifffile.imread(mask_path)
            change_gt = RGB2Index(change_gt, mode='NonZero')  # 把三位的 RGB， 根据 palatte 转换为 index
            self.change_gt = change_gt

    def train(self):
        for idx in range(0, self.num_pairs):
            # get data
            self.load_datapath_change_gt(idx)
            image1 = openImage(self.image_paths[0])
            image2 = openImage(self.image_paths[1])
            tensor1 = self.to_tensor(image1)[:3, :, :]
            tensor2 = self.to_tensor(image2)[:3, :, :]

            self.model.train()
            _, feat1 = self.model(tensor1)
            _, feat2 = self.model(tensor2)

            output = self.model.classifier_cd(torch.concate((feat1, feat2), dim=1))
            output = function.interpolate(output, size=self.model.input_size, mode='bilinear', align_corners=False)
            output = output[:, :, :self.model.input_size[0], :self.model.input_size[1]]

            assert output.size()[2:] == self.change_gt.size()[1:]
            loss = self.criterion(output, self.change_gt)

            # todo: log losses

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.lr_scheduler.step()

    def multi_scale_predict(self, image, flip=False):
        input_size = (image.size(2), image.size(3))
        upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
        total_predictions = np.zeros((self.num_classes, image.size(2), image.size(3)))

        image = image.data.data.cpu().numpy()
        for scale in self.scales:
            scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
            scaled_img = torch.from_numpy(scaled_img).to(self.device)
            scaled_prediction = upsample(self.model(scaled_img).cpu())

            if flip:
                fliped_img = scaled_img.flip(-1).to(self.device)
                fliped_predictions = upsample(self.model(fliped_img).cpu())
                scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
            total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

        total_predictions /= len(self.scales)
        return total_predictions

    def predict(self, tensor):
        self.model.eval()
        inputs = self.normalize(tensor).unsqueeze(0)
        prediction = self.multi_scale_predict(inputs)
        prediction = function.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()

        return prediction

    def detect(self):
        # testing
        save_path = self.config["Output"]["img_outpath"] + '/omit number.txt'
        with open(save_path, 'a') as f:
            f.write('########################################\n')
            f.write('Experiment:{}\n'.format(self.exp))

        with torch.no_grad():
            for index in range(0, self.num_pairs):  # 顺序检索
                # get data
                self.load_datapath_change_gt(index)
                image1 = openImage(self.image_paths[0])
                image2 = openImage(self.image_paths[1])
                tensor1 = self.to_tensor(image1)[:3, :, :]
                tensor2 = self.to_tensor(image2)[:3, :, :]

                # predict the segmentation result based on current load model
                prediction1 = self.predict(tensor1)
                prediction2 = self.predict(tensor2)
                # print('set(prediction1)', np.unique(prediction1))

                for n_seg in self.n_segments:
                    for merge_region in self.merge_regions:
                        # get fused superpixels from images
                        sp_fused, sp1, sp2 = \
                            SP_fusion(image1, image2, n_seg, self.compactness,  # 超像素融合
                                      self.merge, merge_regions=merge_region)

                        if self.record_spAcc:  # record super pixel accuracy
                            self.record_acc(sp_fused, sp1, sp2, prediction1, prediction2, n_seg, merge_region)

                        # 根据分割结果的变化
                        self.change_seg = np.zeros(sp_fused.shape)
                        self.change_seg[prediction1 != prediction2] = 1
                        self.change_seg[prediction1 == 0] = 0  # 缺省类别忽略
                        self.change_seg[prediction2 == 0] = 0

                        # 超像素整形后的变化
                        # change the prediction based on SP reult
                        self.prediction_SP1 = classOfSP(sp_fused, prediction1)
                        self.prediction_SP2 = classOfSP(sp_fused, prediction2)
                        self.change_fusedSP = self.change_detect(self.prediction_SP1, self.prediction_SP2, sp_fused)

                        # 根据超像素融合的结果修改分割变化的变化
                        self.change_pred_change = self.change_detect_pred_change(sp_fused, self.change_seg, prediction1)

                        self.save_results(prediction1, prediction2)

                        self.cal_metrics(change_pred_change=self.change_pred_change, tag='pred')
                        self.cal_metrics(change_pred_change=self.change_seg, tag='seg')
                        self.cal_metrics(change_pred_change=self.change_fusedSP, tag='fused')

        # get metrics together
        def average(val1, val2, val3, val4):
            return [sum(val1) / len(val1), sum(val2) / len(val2), sum(val3) / len(val3), sum(val4) / len(val4)]

        omit_avg_seg_SP, prec_avg_seg_SP, pixAcc_avg_seg_SP, f1_score_seg_SP = average(self.omit_seg_SP,
                                                                                       self.falseAlarm_seg_SP,
                                                                                       self.pixAcc_seg_SP,
                                                                                       self.f1_score_seg_SP)
        omit_avg_seg, prec_avg_seg, pixAcc_avg_seg, f1_score_seg = average(self.omit_seg, self.falseAlarm_seg,
                                                                           self.pixAcc_seg, self.f1_score_seg)
        omit_avg_SP, prec_avg_SP, pixAcc_avg_SP, f1_score_SP = average(self.omit_SP, self.falseAlarm_SP,
                                                                       self.pixAcc_SP, self.f1_score_SP)

        return [omit_avg_seg_SP, omit_avg_seg, omit_avg_SP], \
               [prec_avg_seg_SP, prec_avg_seg, prec_avg_SP], \
               [pixAcc_avg_seg_SP, pixAcc_avg_seg_SP, pixAcc_avg_SP], \
               [f1_score_seg_SP, f1_score_seg, f1_score_SP]

    def change_detect(self, prediction_SP1, prediction_SP2, fused_sp):
        """
        :param fused_sp: the fused super pixel
        :param prediction_SP1: semantic prediction based on 1st image
        :param prediction_SP2: semantic prediction based on 2st image
        self.ignore_pixels: super pixels contain more pixel than this one will be considered
        :return: change based on fused_superpixel prediction
        """
        outset = np.unique(fused_sp.flatten())  # the unique labels
        change_pred = np.zeros(fused_sp.shape)
        for i in outset:
            if len(prediction_SP1[fused_sp == i]) > self.ignore_pixels:
                if prediction_SP1[fused_sp == i][0] != prediction_SP2[fused_sp == i][0] and \
                        prediction_SP2[fused_sp == i][0] != 0 and prediction_SP1[fused_sp == i][0] != 0:
                    change_pred[fused_sp == i] = 1

        return change_pred

    def change_detect_pred_change(self, sp_fused, change_seg, prediction1):
        """
        return numpy array changes
        """
        outset = np.unique(sp_fused.flatten())  # the unique labels
        change_based_pred = np.zeros(sp_fused.shape)
        for i in outset:
            totalpix_sp = len(prediction1[sp_fused == i])
            if totalpix_sp > self.ignore_pixels:
                if sum(change_seg[sp_fused == i]) / totalpix_sp > self.threshold:
                    change_based_pred[sp_fused == i] = 1
        return change_based_pred

    def save_images(self, mask, output_path, image_file, tag):
        """
        save the mask(numpy array) of value [0, classes-1] into os.path.join(output_path, image_file + tag + '.png')
        :param mask: the numpy array that you need to save
        :param output_path:
        :param image_file: derived from which file
        :param tag: the tag of the image
        :return: nothing, the saved color mask in output path
        """
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        image_file = os.path.basename(image_file)
        colorized_mask = colorize_mask(mask, self.palette)
        if "tiff" in image_file:
            colorized_mask.save(os.path.join(output_path, image_file[:-5] + tag + '.png'))
        else:
            colorized_mask.save(os.path.join(output_path, image_file[:-4] + tag + '.png'))

    def record_acc(self, sp_fused, sp1, sp2, prediction1,
                   prediction2, n_seg, merge_region, label1=None, label2=None):
        """
        calculate SP accuracy and record it into ./result.txt
        :return: the saved txt file in the current directory
        """
        # sp accuracy based on label
        # label1_fuse_acc = sp_accuracy(sp_fused, label1)
        # label1_nofuse_acc = sp_accuracy(sp1, label1)
        # label2_fuse_acc = sp_accuracy(sp_fused, label2)
        # label2_nofuse_acc = sp_accuracy(sp2, label2)
        pred1_fuse_acc = sp_accuracy(sp_fused, prediction1)
        pred1_nofuse_acc = sp_accuracy(sp1, prediction1)
        pred2_fuse_acc = sp_accuracy(sp_fused, prediction2)
        pred2_nofuse_acc = sp_accuracy(sp2, prediction2)

        save_path = self.config["Output"]["img_outpath"] + '/result.txt'
        with open(save_path, 'a') as f:
            f.write('########################################\n')
            f.write('Experiment:{}\n'.format(self.exp))

            f.write('Result with n_seg = {}, merge_regions = {} and merge = {}: \n'
                    .format(n_seg, merge_region, self.merge))
            # f.write('\t label1_fuse_acc: {} \n'.format(label1_fuse_acc))
            # f.write('\t label1_nofuse_acc: {} \n'.format(label1_nofuse_acc))
            # f.write('\t label2_fuse_acc: {} \n'.format(label2_fuse_acc))
            # f.write('\t label2_nofuse_acc: {} \n'.format(label2_nofuse_acc))
            f.write('\t pred1_fuse_acc: {} \n'.format(pred1_fuse_acc))
            f.write('\t pred1_nofuse_acc: {} \n'.format(pred1_nofuse_acc))
            f.write('\t pred2_fuse_acc: {} \n'.format(pred2_fuse_acc))
            f.write('\t pred2_nofuse_acc: {} \n'.format(pred2_nofuse_acc))
        print('Successfully write to file ~')

    def cal_metrics(self, change_pred_change, tag):
        # metric
        changes = change_pred_change[self.change_gt != 0]
        nochange = change_pred_change[self.change_gt == 0]
        if len(changes) == 0:
            print('***************************')
            print('The incorrect data path is ', self.image_paths[0])
            return

        correct = (change_pred_change == self.change_gt)
        pixelAcc = sum(sum(correct)) / correct.size

        omit = (len(changes) - sum(changes)) / correct.size
        falseAlarm = sum(nochange) / correct.size

        if omit > 0 and falseAlarm > 0:
            f1_score = 2 * ((falseAlarm * omit) / (falseAlarm + omit))
        else:
            f1_score = 0

        if 'pred' in tag:
            self.pixAcc_seg_SP.append(pixelAcc)
            self.omit_seg_SP.append(omit)
            self.falseAlarm_seg_SP.append(falseAlarm)
            self.f1_score_seg_SP.append(f1_score)
        elif 'seg' in tag:
            self.pixAcc_seg.append(pixelAcc)
            self.omit_seg.append(omit)
            self.falseAlarm_seg.append(falseAlarm)
            self.f1_score_seg.append(f1_score)
        elif 'fused' in tag:
            self.pixAcc_SP.append(pixelAcc)
            self.omit_SP.append(omit)
            self.falseAlarm_SP.append(falseAlarm)
            self.f1_score_SP.append(f1_score)

        save_path = self.config["Output"]["img_outpath"] + '/omit number.txt'
        with open(save_path, 'a') as f:
            f.write('omit: {}, falseAlarm: {}, f1score: {}, pixelAcc: {} \n'.format(
                omit, falseAlarm, f1_score, pixelAcc))

        print('omit: {}, falseAlarm: {}, f1score: {}, pixelAcc: {}'.format(
            omit, falseAlarm, f1_score, pixelAcc))

    def save_results(self, prediction1, prediction2):
        # save prediction result on images
        match_idx = Path(self.image_paths[0])._parts[-2]
        self.out_path_tmp = os.path.join(self.out_path, match_idx)

        self.save_images(prediction1, self.out_path_tmp, self.image_paths[0], 'pred')
        self.save_images(self.prediction_SP1, self.out_path_tmp, self.image_paths[0], 'pred_afterSP')
        self.save_images(prediction2, self.out_path_tmp, self.image_paths[1], 'pred')
        self.save_images(self.prediction_SP2, self.out_path_tmp, self.image_paths[1], 'pred_afterSP')
        self.save_images(self.change_fusedSP, self.out_path_tmp, self.image_paths[0], 'change_SP')
        self.save_images(self.change_gt, self.out_path_tmp, self.image_paths[0], 'change_gt')
        self.save_images(self.change_seg, self.out_path_tmp, self.image_paths[0], 'change_seg')
        self.save_images(self.change_pred_change, self.out_path_tmp, self.image_paths[0], 'change_pred_SP')
        save_path = self.config["Output"]["img_outpath"] + '/omit number.txt'
        with open(save_path, 'a') as f:
            f.write('Success and save result to ' + self.out_path_tmp + '\n')
        print('Success and save result to ', self.out_path_tmp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change Detection arguments')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='the path to the config file')
    args = parser.parse_args()
    config = json.load(open(args.config))

    detector = ChangeDetection(config)
    omit, falseAlarm, pixAcc, f1_score = detector.detect()

    save_path = config["Output"]["img_outpath"] + '/omit number.txt'
    with open(save_path, 'a') as f:
        f.write('change detection: seg_SP, seg, SP')
        f.write('omit: {} \n falseAlarm: {}, \n pixelAcc:{}, \n f1_score:{}'.format(omit[0], falseAlarm[0], pixAcc[0],
                                                                                    f1_score[0]))
        f.write('omit: {} \n falseAlarm: {}, \n pixelAcc:{}, \n f1_score:{}'.format(omit[1], falseAlarm[1], pixAcc[1],
                                                                                    f1_score[1]))
        f.write('omit: {} \n falseAlarm: {}, \n pixelAcc:{}, \n f1_score:{}'.format(omit[2], falseAlarm[2], pixAcc[2],
                                                                                    f1_score[2]))
    print('change detection: seg_SP, seg, SP')
    print('omit: {} \n falseAlarm: {}, \n pixelAcc:{}, \n f1_score:{}'.format(omit[0], falseAlarm[0], pixAcc[0],
                                                                              f1_score[0]))
    print('omit: {} \n falseAlarm: {}, \n pixelAcc:{}, \n f1_score:{}'.format(omit[1], falseAlarm[1], pixAcc[1],
                                                                              f1_score[1]))
    print('omit: {} \n falseAlarm: {}, \n pixelAcc:{}, \n f1_score:{}'.format(omit[2], falseAlarm[2], pixAcc[2],
                                                                              f1_score[2]))

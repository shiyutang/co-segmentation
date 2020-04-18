import os
import sys
import math
import copy
import zipfile
import logging
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
from skimage import io
from scipy import ndimage
from itertools import chain
from collections import Counter
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
__all__ = ['ResNet','resnet152', 'BasicBlock', 'Bottleneck']
model_urls = { 'resnet152': 'https://hangzh.s3.amazonaws.com/encoding/models/resnet152-0d43d698.zip'}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert(len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i]+y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, dilated=True, multi_grid=False,
                 deep_base=True, norm_layer=nn.BatchNorm2d):
        self.inplanes = 128 if deep_base else 64
        super(ResNet, self).__init__()
        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            if multi_grid:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer,
                                               multi_grid=True)
            else:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        multi_dilations = [4, 8, 16]
        if multi_grid:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilations[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if multi_grid:
                layers.append(block(self.inplanes, planes, dilation=multi_dilations[i],
                                    previous_dilation=dilation, norm_layer=norm_layer))
            else:
                layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet152(pretrained=False, root='./pretrained', **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet152'], model_dir=root))
    return model


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1].split('.')[0]
    cached_file = os.path.join(model_dir, filename+'.pth')
    if not os.path.exists(cached_file):
        cached_file = os.path.join(model_dir, filename+'.zip')
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
        zip_ref = zipfile.ZipFile(cached_file, 'r')
        zip_ref.extractall(model_dir)
        zip_ref.close()
        os.remove(cached_file)
        cached_file = os.path.join(model_dir, filename+'.pth')
    return torch.load(cached_file, map_location=map_location)


def summary(model, input_shape, batch_size=-1, intputshow=True):
    '''

    :param model:
    :param input_shape:
    :param batch_size:
    :param intputshow:
    :return: model infor
    '''
    def register_hook(module):
        def hook(module, input, output=None):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)
                    and not (module == model)) and 'torch' in str(module.__class__):
            if intputshow is True:
                hooks.append(module.register_forward_pre_hook(hook))
            else:
                hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)
    model(torch.zeros(input_shape))

    # remove these hooks
    for h in hooks:
        h.remove()

    model_info = ''
    model_info += "-----------------------------------------------------------------------\n"
    line_new = "{:>25}  {:>25} {:>15}".format("Layer (type)", "Input Shape", "Param #")
    model_info += line_new + '\n'
    model_info += "=======================================================================\n"

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        line_new = "{:>25}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        if intputshow is True:
            total_output += np.prod(summary[layer]["input_shape"])
        else:
            total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]

        model_info += line_new + '\n'

    model_info += "=======================================================================\n"
    model_info += "Total params: {0:,}\n".format(total_params)
    model_info += "Trainable params: {0:,}\n".format(trainable_params)
    model_info += "Non-trainable params: {0:,}\n".format(total_params - trainable_params)
    model_info += "-----------------------------------------------------------------------\n"
    return model_info


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'


def set_trainable_attr(m,b):
    m.trainable = b
    for p in m.parameters(): p.requires_grad = b


def apply_leaf(m, f):
    c = m if isinstance(m, (list, tuple)) else list(m.children())
    if isinstance(m, nn.Module):
        f(m)
    if len(c)>0:
        for l in c:
            apply_leaf(l,f)


def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, norm_layer)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), out_channels,
                      kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class PSPNet(BaseModel):
    def __init__(self, num_classes=8, in_channels=3, backbone='resnet152', pretrained=True, use_aux=True,
                 freeze_bn=False, freeze_backbone=False):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        model = resnet152(pretrained=False, norm_layer=norm_layer, )
        m_out_sz = model.fc.in_features
        self.use_aux = use_aux

        self.initial = nn.Sequential(*list(model.children())[:4])
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.master_branch = nn.Sequential(
            _PSPModule(m_out_sz, bin_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1)
        )

        self.auxiliary_branch = nn.Sequential(
            nn.Conv2d(m_out_sz // 2, m_out_sz // 4, kernel_size=3, padding=1, bias=False),
            norm_layer(m_out_sz // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(m_out_sz // 4, num_classes, kernel_size=1)
        )

        initialize_weights(self.master_branch, self.auxiliary_branch)
        if freeze_bn: self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)

        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear')
        output = output[:, :, :input_size[0], :input_size[1]]

        if self.training and self.use_aux:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear')
            aux = aux[:, :, :input_size[0], :input_size[1]]
            return output, aux
        return output

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(),
                     self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img


def multi_scale_predict(model, image, scales, num_classes, device, flip=False):
    input_size = (image.size(2), image.size(3))
    upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

    image = image.data.data.cpu().numpy()
    for scale in scales:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
        scaled_img = torch.from_numpy(scaled_img).to(device)
        scaled_prediction = upsample(model(scaled_img).cpu())

        if flip:
            fliped_img = scaled_img.flip(-1).to(device)
            fliped_predictions = upsample(model(fliped_img).cpu())
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions


def save_images(mask, output_path, image_file, tag, palette):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    image_file = os.path.basename(image_file).split('.')[0]
    colorized_mask = colorize_mask(mask, palette)
    colorized_mask.save(os.path.join(output_path, image_file + tag + '.png'))


def SP_fusion(image1, image2, n_segments, compactness, merge, merge_regions=50, img_names = ('','')):
    '''
    :param image1: image of time1
    :param image2: image of time2
    :param n_segments:
    :param compactness:
    :return: the fused superpixel of image1 and image2
    '''
    from skimage.segmentation import slic, mark_boundaries
    import matplotlib.pyplot as plt

    ## SLIC Superpixel and save
    labels1 = slic(np.array(image1), n_segments, compactness)
    labels2 = slic(np.array(image2), n_segments, compactness)

    result_nomerge = [labels1, labels2]
    if merge:
        import skimage.external.tifffile as tifffile
        from os.path import abspath, dirname
        Maindirc = abspath(dirname(__file__))

        # set constant for superpixels merge
        sys.path.insert(0, 'MergeTool')

        MergeTool = 'SuperPixelMerge.exe'
        SP_label = Maindirc + '/Superpixel.tif'
        SPMG_label = Maindirc + '/Merge.tif'
        MG_Criterion = 3  # 0, 1, 2, 3
        Num_of_Region = merge_regions  # the number of regions after region merging
        MG_Shape = 0.7
        MG_Compact = 0.7

        result = []
        for img_name, labels in zip(img_names, [labels1, labels2]):
            ## SLIC Superpixel and save
            labels = labels.astype('int32')
            tifffile.imsave('Superpixel.tif', labels, photometric='minisblack')

            ## Call the Superpixel Merge tool, format the command line input
            os.chdir('/data/Project_prep/superpixel-cosegmentation/MergeTool/')
            cmd_line = '{} {} {} {} {} {} {} {} {}'.format(MergeTool, img_name, SP_label, SPMG_label, \
                                                           MG_Criterion, Num_of_Region, MG_Shape, ' ', MG_Compact)
            print('cmd_line', cmd_line)
            os.system(cmd_line)  # call the Superpixel Merge Tool
            os.chdir('..')

            ## save merged slic labels
            MG_labels = tifffile.imread(SPMG_label)
            result.append(MG_labels)

        labels1, labels2 = result

    fusion_labels = labels1+labels2

    # test
    # out1 = mark_boundaries(image1, result_nomerge[0])
    # io.imshow(out1)
    # plt.show()
    # out = mark_boundaries(image1, result[0])  # this shows previous result stored on PC
    # io.imshow(out)
    # plt.show()
    outf = mark_boundaries(image1, fusion_labels, mode='inner')
    # out1 = mark_boundaries(image1, labels1)
    # out2 = mark_boundaries(image2, labels2)
    # io.imshow(out1)
    # plt.show()
    # io.imshow(out2)
    # plt.show()
    io.imshow(outf)
    plt.show()

    return fusion_labels, labels1, labels2


def classOfSP(sp, prediction):
    '''
    :param sp: super pixel label of a image | type: <numpy.ndarray>
    :param prediction: the probability of segmented result | type: list 200*200
    :return: the segmented result as superpixel | type: list
    '''
    outset = np.unique(sp.flatten())  # the unique labels
    fuse_prediction = copy.deepcopy(prediction)
    for i in outset:
        mostpred, times = Counter(prediction[sp == i]).most_common(1)[0]
        fuse_prediction[sp == i] = mostpred

    # test
    # print('fuse_prediction', fuse_prediction)

    return fuse_prediction


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


def accuracy(sp, label):
    """
    :param sp:
    :param label:
    :return: the accuracy of superpixel segmentation
    """
    if len(label.shape) == 3:
        label = label2intarray(label)
    sp_pred = classOfSP(sp, label)
    correct = sp_pred[sp_pred == label]
    acc = len(correct)/(sp_pred.size)

    return round(acc, 3)


def main():
    # Dataset used for training the model
    scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.45734706, 0.43338275, 0.40058118], [0.23965294, 0.23532275, 0.2398498])
    num_classes = 8
    palette = [0, 0, 0, 150, 250, 0, 0, 250, 0, 0, 100, 0, 200,
               0, 0, 255, 255, 255, 0, 0, 200, 0, 150, 250]

    def predict(image):
        input = normalize(to_tensor(image)).unsqueeze(0)
        prediction = multi_scale_predict(model, input, scales, num_classes, device)
        prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()

        return prediction

    def change_detect(sp_fused, prediction1, prediction2, ignore_pixels=20):
        outset = np.unique(sp_fused.flatten())  # the unique labels
        change = np.zeros(sp_fused.shape)
        for i in outset:
            if len(prediction1[sp_fused == i]) > ignore_pixels:
                if prediction1[sp_fused == i][0] != prediction2[sp_fused == i][0]:
                    change[sp_fused == i] = 1
            # test
            # if i == 0:
            #     print('sp_fused == i', sp_fused == i)  # true/False list
            #     print('prediction1[sp_fused==i]', prediction1[sp_fused == i])  # [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ]
            #     print('change', change)

        return change

    def change_detect_pred_change(pred_change, sp_fused, threshold=0.7, ignore_pixels=20):
        outset = np.unique(sp_fused.flatten())  # the unique labels
        change = np.zeros(sp_fused.shape)
        for i in outset:
            totalpix_sp = len(prediction1[sp_fused == i])
            if totalpix_sp > ignore_pixels:
                if sum(pred_change[sp_fused == i])/totalpix_sp > threshold:
                    change[sp_fused == i] = 1
        return change

    # Model
    model = PSPNet()
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    checkpoint = torch.load('./best_model.pth', map_location = device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    img_ext, lbl_ext = 'jpg', 'png'
    image_files = sorted(glob(os.path.join('test', f'*.{img_ext}')))
    label_files = sorted(glob(os.path.join('label', f'*.{lbl_ext}')))

    with torch.no_grad():
        for index in tqdm(range(len(image_files)//2)):
            # for each pic in image_files, it will generate classes for each superpixel
            image1 = Image.open(image_files[2*index]).convert('RGB')
            image2 = Image.open(image_files[2*index+1]).convert('RGB')
            label1 = np.array(Image.open(label_files[2*index]).convert('RGB'))
            label2 = np.array(Image.open(label_files[2*index+1]).convert('RGB'))

            prediction1 = predict(image1)
            prediction2 = predict(image2)

            merge = False
            n_segments, compactness, merge_regions = [150], 10, [0]  # 50, 150, 500,
            if merge:
                n_segments, merge_regions = [1000, 1500, 2000, 3000], [100, 150, 200]
            for n_seg in n_segments:
                for merge_region in merge_regions:
                    # get fused superpixels from images
                    sp_fused, sp1, sp2 = SP_fusion(image1, image2, n_seg, compactness, merge, merge_regions=merge_region)

                    # change the prediction
                    prediction_SP1 = classOfSP(sp_fused, prediction1)  # the prediction for fused_SP = int(classes) array
                    prediction_SP2 = classOfSP(sp_fused, prediction2)

                    # label1_fuse_acc = accuracy(sp_fused, label1)
                    # label1_nofuse_acc = accuracy(sp1, label1)
                    # label2_fuse_acc = accuracy(sp_fused, label2)
                    # label2_nofuse_acc = accuracy(sp2, label2)
                    # pred1_fuse_acc = accuracy(sp_fused, prediction1)
                    # pred1_nofuse_acc = accuracy(sp2, prediction1)
                    # pred2_fuse_acc = accuracy(sp_fused, prediction2)
                    # pred2_nofuse_acc = accuracy(sp2, prediction2)
                    # save_path = './result.txt'
                    # with open(save_path, 'a') as f:
                    #     f.write('Result with n_seg = {}, merge_regions = {} and merge = {}: \n'
                    #             .format(n_seg, merge_region, merge))
                    #     f.write('\t label1_fuse_acc: {} \n'.format(label1_fuse_acc))
                    #     f.write('\t label1_nofuse_acc: {} \n'.format(label1_nofuse_acc))
                    #     f.write('\t label2_fuse_acc: {} \n'.format(label2_fuse_acc))
                    #     f.write('\t label2_nofuse_acc: {} \n'.format(label2_nofuse_acc))
                    #     f.write('\t pred1_fuse_acc: {} \n'.format(pred1_fuse_acc))
                    #     f.write('\t pred1_nofuse_acc: {} \n'.format(pred1_nofuse_acc))
                    #     f.write('\t pred2_fuse_acc: {} \n'.format(pred2_fuse_acc))
                    #     f.write('\t pred2_nofuse_acc: {} \n'.format(pred2_nofuse_acc))
                    # print('Successfully write to file ~')

                    change = change_detect(sp_fused, prediction_SP1, prediction_SP2)  # the change based co-seg
                    change_seg, change_gt = np.zeros(sp_fused.shape),np.zeros(sp_fused.shape)
                    change_seg[prediction1 != prediction2] = 1                        # the change based prediction
                    change_pred_change = change_detect_pred_change(change_seg, sp_fused)
                    label1, label2 = label2intarray(label1), label2intarray(label2)
                    change_gt[label1 != label2] = 1                # the change based labels

                    save_images(prediction1, 'outputs', image_files[2*index], 'pred', palette)
                    save_images(prediction_SP1, 'outputs', image_files[2*index], 'pred_afterSP', palette)
                    save_images(prediction2, 'outputs', image_files[2*index+1], 'pred', palette)
                    save_images(prediction_SP2, 'outputs', image_files[2*index+1], 'pred_afterSP', palette)
                    save_images(change, 'outputs', image_files[2*index], 'change', palette)
                    save_images(change_gt, 'outputs', image_files[2*index], 'change_gt', palette)
                    save_images(change_seg, 'outputs', image_files[2*index], 'change_seg', palette)
                    save_images(change_pred_change, 'outputs', image_files[2*index], 'change_pred_change', palette)


if __name__ == '__main__':
    main()


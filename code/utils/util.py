import argparse
import random
from typing import Iterable, Union
from copy import deepcopy as dcopy
from typing import List, Set
import collections
from functools import partial, reduce
import torch
import numpy as np
import os
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings
import torch.nn as nn

##### Hyper Parameters Define #####

def _parser_(input_strings: str) -> Union[dict, None]:
    if input_strings.__len__() == 0:
        return None
    assert input_strings.find('=') > 0, f"Input args should include '=' to include value"
    keys, value = input_strings.split('=')[:-1][0].replace(' ', ''), input_strings.split('=')[1].replace(' ', '')
    keys = keys.split('.')
    keys.reverse()
    for k in keys:
        d = {}
        d[k] =value
        value = dcopy(d)
    return dict(value)

def _parser(strings: List[str]) -> List[dict]:
    assert isinstance(strings, list)
    args: List[dict] = [_parser_(s) for s in strings]
    args = reduce(lambda x, y: dict_merge(x, y, True), args)
    return args

def yaml_parser() -> dict:
    parser = argparse.ArgumentParser('Augmnet oarser for yaml config')
    parser.add_argument('strings', nargs='*', type=str, default=[''])
    parser.add_argument("--local_rank", type=int)
    #parser.add_argument('--var', type=int, default=24)
    #add args.variable here
    args: argparse.Namespace = parser.parse_args()
    args: dict = _parser(args.strings)
    return args

def dict_merge(dct: dict, merge_dct: dict, re=False):
    '''
    Recursive dict merge. Instead updating only top-level keys, dict_merge recuses down into dicts nested
    to an arbitrary depth, updating keys. The ""merge_dct"" is merged into "dct".
    '''
    if merge_dct is None:
        if re:
            return dct
        else:
            return 
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct(k), collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            try:
                dct[k] = type(dct[k])(eval(merge_dct[k])) if type(dct[k]) in (bool, list) else type(dct[k])(
                    merge_dct[k])
            except:
                dct[k] = merge_dct[k]
    if re:
        return dcopy(dct)

##### Timer ######
def now_time():
    time = datetime.datetime.now()
    return str(time)[:19]

##### Progress Bar #####

tqdm_ = partial(tqdm, ncols=125, leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')

##### Coding #####
def class2one_hot(seg: torch.Tensor, num_class: int) -> torch.Tensor:
    '''
    [b, w, h] containing (0, 1, ..., c) -> [b, c, w, h] containing (0, 1)
    '''
    if len(seg.shape) == 2:
        seg = seg.unsqueeze(dim=0) # Must 3 dim
    if len(seg.shape) == 4:
        seg = seg.squeeze(dim=1)
    assert sset(seg, list(range(num_class))), 'The value of segmentation outside the num_class!'
    b, w, h = seg.shape # Tuple [int, int, int]
    res = torch.stack([seg == c for c in range(num_class)], dim=1).type(torch.int32)
    assert res.shape == (b, num_class, w, h)
    assert one_hot(res)
    
    return res 

def probs2class(probs: torch.Tensor) -> torch.Tensor:
    '''
    [b, c, w, h] containing(float in range(0, 1)) -> [b, w, h] containing ([0, 1, ..., c])
    '''
    b, _, w, h = probs.shape
    assert simplex(probs), '{} is not a probability'.format(probs)
    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res

def probs2one_hot(probs: torch.Tensor) -> torch.Tensor:
    _, num_class, _, _  = probs.shape
    assert simplex(probs), '{} is not a probability'.format(probs)
    res = class2one_hot(probs2class(probs), num_class)
    assert res.shape == probs.shape
    assert one_hot(res)
    return res

def label_onehot(inputs, num_class):
    '''
    inputs is class label
    return one_hot label 
    dim will be increasee
    '''
    batch_size, image_h, image_w = inputs.shape
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_class, image_h, image_w]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)


def uniq(a: torch.Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: torch.Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub) 

def simplex(t: torch.Tensor, axis=1) -> bool:
    '''
    Check if the maticx is the probability in axis dimension.
    '''
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def one_hot(t: torch.Tensor, axis=1) ->  bool:
    '''
    Check if the Tensor is One-hot coding
    '''
    return simplex(t, axis) and sset(t, [0, 1])

def intersection(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    '''
    a and b must only contain 0 or 1, the function compute the intersection of two tensor.
    a & b
    '''
    assert a.shape == b.shape, '{}.shape must be the same as {}'.format(a, b)
    assert sset(a, [0, 1]), '{} must only contain 0, 1'.format(a)
    assert sset(b, [0, 1]), '{} must only contain 0, 1'.format(b)
    return a & b

class iterator_(object):
    def __init__(self, dataloader: DataLoader) -> None:
        super().__init__()
        self.dataloader = dcopy(dataloader)
        self.iter_dataloader = iter(dataloader)
        self.cache = None

    def __next__(self):
        try:
            self.cache = self.iter_dataloader.__next__()
            return self.cache
        except StopIteration:
            self.iter_dataloader = iter(self.dataloader)
            self.cache = self.iter_dataloader.__next__()
            return self.cache
    def __cache__(self):
        if self.cache is not None:
            return self.cache
        else:
            warnings.warn('No cache found ,iterator forward')
            return self.__next__()

def apply_dropout(m):
    if type(m) == nn.Dropout2d:
        m.train()

##### Scheduler #####
class RampUpScheduler():
    def __init__(self, begin_epoch, max_epoch, max_value, ramp_mult):
        super().__init__()
        self.begin_epoch = begin_epoch
        self.max_epoch = max_epoch
        self.ramp_mult = ramp_mult
        self.max_value = max_value
        self.epoch = 0
    
    def step(self):
        self.epoch += 1

    @property
    def value(self):
        return self.get_lr(self.epoch, self.begin_epoch, self.max_epoch, self.max_value,self.ramp_mult)

    def get_lr(self, epoch, begin_epoch, max_epochs, max_val, mult):
        if epoch < begin_epoch:
            return 0.
        elif epoch >= max_epochs:
            return max_val
        return max_val * np.exp(mult * (1 - float(epoch - begin_epoch) / (max_epochs - begin_epoch)) ** 2 )


##### Compute mIoU #####
def mask_label(label, mask):
    '''
    label is the original label (contains -1), mask is the valid region in pseudo label (type=long)
    return a label with invalid region = -1
    '''
    label_tmp = label.clone()
    mask_ = (1 - mask.float()).bool()
    label_tmp[mask_] = -1
    return label_tmp.long()


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    Warning: torch.distributed.all_gather has no gradient.
    """
    tensor = tensor.contiguous()
    tensor_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensor_gather, tensor, async_op=False)
    output = torch.cat(tensor_gather, dim=0)

    return output

def converToSlice(input):

    D = input.size(-1)
    input2d = input[..., 0]
    for i in range(1, D):
        input2dtmp = input[..., i]
        input2d = torch.cat((input2d, input2dtmp), dim=0)

    return input2d

def rand_bbox_3d(size, lam=None):
    # past implementation
    W = size[2]
    H = size[3]
    D = size[4]
    B = size[0]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cut_z = int(D * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W / 8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H / 8), high=H)
    cz = np.random.randint(size=[B, ], low=int(D / 8), high=D)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbz1 = np.clip(cz - cut_z // 2, 0, D)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbz2 = np.clip(cz + cut_z // 2, 0, D)

    return bbx1, bby1, bbz1,bbx2, bby2, bbz2


def cut_mix(unlabeled_image=None, unlabeled_mask=None,mask=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_mask= mask.clone()
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]].cuda()
    u_bbx1, u_bby1,u_bbz1, u_bbx2, u_bby2, u_bbz2 = rand_bbox_3d(unlabeled_image.size(), lam=np.random.beta(4, 4))

    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

        mix_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

    del unlabeled_image, unlabeled_mask, mask

    return mix_unlabeled_image, mix_unlabeled_target, mix_mask


def rand_bbox_2d(size, lam=None):
    # past implementation
    W = size[2]
    H = size[3]
    B = size[0]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W / 8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H / 8), high=H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cut_mix_2d(unlabeled_image=None, unlabeled_mask=None, mask=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_mask = mask.clone()
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]].cuda()
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox_2d(unlabeled_image.size(), lam=np.random.beta(4, 4))

    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del unlabeled_image, unlabeled_mask, mask

    return mix_unlabeled_image, mix_unlabeled_target, mix_mask


#   add noise ###
import cv2
import numpy as np
from scipy.ndimage import convolve1d
import scipy.ndimage as ndimage
from scipy.ndimage.measurements import center_of_mass


def blur_mri_tensor(mri_tensor, kernel_size=(5, 5), sigma=0):
    """
    Blurs an MRI tensor using Gaussian blur.

    Args:
        mri_tensor (ndarray): MRI tensor to be blurred. H,W
        kernel_size (tuple): Size of the Gaussian kernel (default: (5, 5)).
        sigma (float): Standard deviation of the Gaussian kernel (default: 0).

    Returns:
        ndarray: Blurred MRI tensor.
    """
    # Convert MRI tensor to image format (e.g., grayscale)
    mri_image = np.squeeze(mri_tensor)  # Remove single-dimensional axes if present
    mri_image = (mri_image - np.min(mri_image)) / (np.max(mri_image) - np.min(mri_image))
    mri_image = mri_image * 255.0
    mri_image = np.uint8(mri_image)  # Convert to 8-bit unsigned integer

    # Apply padding on image
    padding = (kernel_size[0] -1 )//2
    mri_image = np.pad(mri_image, ((padding, padding), (padding, padding)), 'reflect')

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(mri_image, kernel_size, sigma)
    blurred_image = blurred_image / 255.0

    # Apply cropping on processed image
    blurred_image = blurred_image[padding:-padding, padding:-padding]

    return blurred_image


def blur_mri_volume(mri_volume, kernel_size=(5, 5), sigma=0):
    """
    Blurs an MRI volume using Gaussian blur.

    Args:
        mri_volume (ndarray): MRI volume to be blurred, with shape (height, width, depth).
        kernel_size (tuple): Size of the Gaussian kernel (default: (5, 5)).
        sigma (float): Standard deviation of the Gaussian kernel (default: 0).

    Returns:
        ndarray: Blurred MRI volume.
    """
    blurred_slices = []

    # Iterate over each slice along the depth dimension
    for slice_idx in range(mri_volume.shape[2]):
        # Extract the slice
        mri_slice = mri_volume[:, :, slice_idx]

        # Blur the slice using the previous blur function
        blurred_slice = blur_mri_tensor(mri_slice, kernel_size, sigma)
        # blurred_slice = separable_gaussian_filter(mri_slice, sigma)

        # Append the blurred slice to the list
        blurred_slices.append(blurred_slice)

    # Stack the blurred slices along the depth dimension to form the blurred volume
    blurred_volume = np.stack(blurred_slices, axis=2)

    return blurred_volume



def separable_gaussian_filter(image, sigma=0.2):
    mri_image = np.squeeze(image)  # Remove single-dimensional axes if present
    mri_image = (mri_image - np.min(mri_image)) / (np.max(mri_image) - np.min(mri_image))
    mri_image = mri_image * 255.0
    mri_image = np.uint8(mri_image)  # Convert to 8-bit unsigned integer
    # Create 1D Gaussian kernel
    size = int(4 * sigma + 0.5)
    x = np.linspace(-size, size, 2*size + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)

    # Apply horizontal and vertical filtering
    filtered_image = convolve1d(mri_image, kernel, axis=1)
    filtered_image = convolve1d(filtered_image, kernel, axis=0)

    return filtered_image


def add_speckle_noise(data, mean=0, std=0.7):
    """
    Adds speckle noise to the input data.

    Args:
        data (ndarray): Input data.
        mean (float): Mean of the noise distribution (default: 0).
        std (float): Standard deviation of the noise distribution (default: 1).

    Returns:
        ndarray: Data with speckle noise added.
    """
    noise = np.random.normal(mean, std, size=data.shape)
    noisy_data = data + data * noise

    return noisy_data




def simulate_motion_blur(image, kernel_size, angle):
    """
    Simulates motion blur in an image.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        kernel_size (int): Size of the motion blur kernel.
        angle (float): Angle of motion blur in degrees.

    Returns:
        numpy.ndarray: Image with motion blur applied as a NumPy array.
    """
    # Convert angle from degrees to radians
    angle_rad = np.deg2rad(angle)

    # Define the motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.cos(angle_rad) / kernel_size
    kernel[:, int((kernel_size - 1) / 2)] = np.sin(angle_rad) / kernel_size

    # Apply padding on image
    padding = (kernel_size - 1) // 2
    image = np.pad(image,((padding,padding),(padding,padding)),'constant',constant_values = (0,0))

    # Apply motion blur using the convolution operation
    blurred_image = ndimage.convolve(image, kernel)

    return blurred_image



def motion_blur_mri_volume(mri_volume, kernel_size=3, angle=5):
    """
    Blurs an MRI volume using Gaussian blur.

    Args:
        mri_volume (ndarray): MRI volume to be blurred, with shape (height, width, depth).
        kernel_size (int): Size of the Gaussian kernel (default: 5).
        sigma (float): Standard deviation of the Gaussian kernel (default: 0).

    Returns:
        ndarray: Blurred MRI volume.
    """
    blurred_slices = []

    # Iterate over each slice along the depth dimension
    for slice_idx in range(mri_volume.shape[2]):
        # Extract the slice
        mri_slice = mri_volume[:, :, slice_idx]

        # Blur the slice using the previous blur function
        blurred_slice = simulate_motion_blur(mri_slice, kernel_size, angle)
        # blurred_slice = separable_gaussian_filter(mri_slice, sigma)

        # Append the blurred slice to the list
        blurred_slices.append(blurred_slice)

    # Stack the blurred slices along the depth dimension to form the blurred volume
    blurred_volume = np.stack(blurred_slices, axis=2)

    return blurred_volume


### ST++: utils.py #################
import numpy as np
from PIL import Image


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iu, np.nanmean(iu)


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    return cmap



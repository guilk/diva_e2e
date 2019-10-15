"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import torch

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch


import numpy as np
import random
import time
import pdb
import os
import functools
from PIL import Image

def flow_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        flow_x_path = os.path.join(video_dir_path, 'flow_x_{:05d}.jpg'.format(i))
        flow_y_path = os.path.join(video_dir_path, 'flow_y_{:05d}.jpg'.format(i))

        if os.path.exists(flow_x_path) and os.path.exists(flow_y_path):
            flow_x = np.array(image_loader(flow_x_path))
            flow_y = np.array(image_loader(flow_y_path))
            img = np.asarray([flow_x, flow_y]).transpose([1,2,0])
            img = Image.fromarray(np.uint8(img))
            video.append(img)
        else:
            return video
    return video

def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for frame_idx in frame_indices:
        # image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        frame_path = os.path.join(video_dir_path, 'image_' + str(frame_idx).zfill(5) + '.jpg')
        if os.path.exists(frame_path):
            video.append(image_loader(frame_path))
        else:
            return video
    return video


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

sss = cfg.SPATIAL_SIZE
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            w, h = img.size
            if (w <= h and w == sss) or (h <= w and h == sss):
                return img.convert('RGB')
            if w < h:
                ow = sss
                oh = int(sss * h / w)
                return img.convert('RGB').resize((ow, oh))
            else:
                oh = sss
                ow = int(sss * w / h)
                return img.convert('RGB').resize((ow, oh))

def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

# sss = 112
def pil_flow_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            w, h = img.size
            if (w <= h and w == sss) or (h <= w and h == sss):
                return img.convert('L')
            if w < h:
                ow = sss
                oh = int(sss * h / w)
                return img.convert('L').resize((ow, oh))
            else:
                oh = sss
                ow = int(sss * w / h)
                return img.convert('L').resize((ow, oh))

def get_default_video_loader(flow=False):
    if flow:
        print('use Flow loader')
        return functools.partial(flow_loader, image_loader=pil_flow_loader)
    else:
        print('use RGB loader')
        image_loader = get_default_image_loader()
        return functools.partial(video_loader, image_loader=image_loader)

class roibatchLoader(data.Dataset):
    def __init__(self, roidb, phase='train', spatial_transform=None, temporal_transform=None, target_transform=None, flow=False):
        self._roidb = roidb
        self.max_num_box = cfg.MAX_NUM_GT_TWINS
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.phase = phase
        self.loader = get_default_video_loader(flow)
        self.video_length = cfg.TRAIN.LENGTH[0]

    def __getitem__(self, index):
        # get the anchor index for current sample index
        item = self._roidb[index]

        # loading frames based on frame indices
        video_info = item['frames'][0]
        start_frame = video_info[1]
        end_frame = video_info[2]
        step = video_info[3] if cfg.INPUT=='video' else 1
        prefix = item['fg_name'] if video_info[0] else item['bg_name']
        frame_indices = range(start_frame, end_frame, step)
        frame_indices = [frame_idx+1 for frame_idx in frame_indices] # ffmpeg decoding from 1 instead of 0

        # padding with the last frame index
        if len(frame_indices) < self.video_length:
            frame_indices = frame_indices + [frame_indices[-1]]*(self.video_length - len(frame_indices))

        # Temporal transform not implemented
        if self.temporal_transform is not None:
            self.temporal_transform.randomize_parameters()
            frame_indices = self.temporal_transform(frame_indices)

        # Here we do not use flipped images as we already use random horizontal flip
        clip = self.loader(prefix, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        data = torch.stack(clip, 0).permute(1,0,2,3)

        if self.phase == 'train':
            # gt windows: (x1, x2, cls)
            gt_inds = np.where(item['gt_classes'] != 0)[0]
            gt_windows = np.empty((len(gt_inds), 3), dtype=np.float32)
            gt_windows[:, 0:2] = item['wins'][gt_inds, :]
            gt_windows[:, -1] = item['gt_classes'][gt_inds]
        else:
            gt_windows = np.zeros((1,3), dtype=np.float32)

        gt_windows = torch.from_numpy(gt_windows)
        gt_windows_padding = gt_windows.new(self.max_num_box, gt_windows.size(1)).zero_()
        num_gt = min(gt_windows.size(0), self.max_num_box)
        gt_windows_padding[:num_gt, :] = gt_windows[:num_gt]

        if self.phase != 'train':
            video_info=''
            for key, value in item.items():
                video_info = video_info + " {}: {}\n".format(key, value)
            video_info = video_info[:-1]
            return data, gt_windows_padding, num_gt, video_info
        else:
            return data, gt_windows_padding, num_gt

    def __len__(self):
        return len(self._roidb)

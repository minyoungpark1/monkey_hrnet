#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:09:14 2019

@author: minyoungpark
"""

from __future__ import print_function, division
import _init_paths
import copy
import logging
import random

import torch
import numpy as np
from torch.utils.data import Dataset

import cv2
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints


logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    
    def __init__(self, cfg, df, is_train=True, transform=None):
        
        self.num_joints = 21
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.db= df
        self.is_train = is_train

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform

        
    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        
        input = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        
        if self.color_rgb:
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

        if input is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))
            
#        data_numpy = np.transpose(data_numpy, (2, 0, 1))
        
        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

#        c = db_rec['center']
#        s = db_rec['scale']
#        score = db_rec['score'] if 'score' in db_rec else 1
#        r = 0
        
#        sf = self.scale_factor
#        rf = self.rotation_factor
#        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
#        r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
#            if random.random() <= 0.6 else 0
#        
        if self.flip and random.random() <= 0.5:
            input = input[:, ::-1, :] - np.zeros_like(input)
            joints, joints_vis = fliplr_joints(
                joints, joints_vis, input.shape[1], self.flip_pairs)
#            c[0] = data_numpy.shape[1] - c[0] - 1

        
#        trans = get_affine_transform(c, s, r, self.image_size)
#        input = cv2.warpAffine(
#            data_numpy,
#            trans,
#            (int(self.image_size[0]), int(self.image_size[1])),
#            flags=cv2.INTER_LINEAR)

#        input = torch.from_numpy(np.flip(data_numpy ,axis=0).copy())
        
        if self.transform:
            input = self.transform(input)
#             input = self.transform(input).half()

#        for i in range(self.num_joints):
#            if joints_vis[i, 0] > 0.0:
#                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        
        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'joints': joints,
            'joints_vis': joints_vis,
#            'center': c,
#            'scale': s,
#            'rotation': r,
#            'score': score
        }

        return input, target, target_weight, meta
    
    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
#         target_weight = np.ones((self.num_joints, 1), dtype=np.float16)
        target_weight = np.ones((self.num_joints, 1))
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]))
#                               dtype=np.float16)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
#                 x = np.arange(0, size, 1, np.float16)
                x = np.arange(0, size, 1)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

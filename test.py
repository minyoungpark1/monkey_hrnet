#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:11:22 2019

@author: minyoungpark
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

# import _init_paths
import sys
sys.path.append('../monkey_hrnet/hrnet/lib/')

from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

from function import train
from function import validate

# import dataset
import models
# torch.set_default_dtype(torch.float16)
parser = argparse.ArgumentParser(description='Train keypoints network')
# general
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)

parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)

# philly
parser.add_argument('--modelDir',
                    help='model directory',
                    type=str,
                    default='')
parser.add_argument('--logDir',
                    help='log directory',
                    type=str,
                    default='')
parser.add_argument('--dataDir',
                    help='data directory',
                    type=str,
                    default='')
parser.add_argument('--prevModelDir',
                    help='prev Model directory',
                    type=str,
                    default='')

#args = parser.parse_args()
args = parser.parse_args(['--cfg', '/home/myp7435/monkey_hrnet/hrnet/experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml'])
args = parser.parse_args(['--modelDir', '/home/myp7435/monkey_hrnet/hrnet/output/coco/pose_hrnet/w32_384x288_adam_lr1e-3/model_best.pth'])

update_config(cfg, args)

logger, final_output_dir, tb_log_dir = create_logger(
    cfg, args.cfg, 'train')

logger.info(pprint.pformat(args))
logger.info(cfg)

# cudnn related setting
cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


# model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=True).half()
model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=True)



# copy model file
try:
#     this_dir = os.path.dirname(__file__)
    this_dir = os.getcwd()
except NameError:  # We are the main py2exe script, not a module
    import sys
    this_dir = os.path.abspath(sys.argv[0])

shutil.copy2(
    os.path.join(this_dir, 'hrnet/lib/models', cfg.MODEL.NAME + '.py'),
    final_output_dir)
# logger.info(pprint.pformat(model))

writer_dict = {
    'writer': SummaryWriter(log_dir=tb_log_dir),
    'train_global_steps': 0,
    'valid_global_steps': 0,
}

# dump_input = torch.rand(
#     (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
# )
# writer_dict['writer'].add_graph(model, (dump_input, ))


# logger.info(get_model_summary(model, dump_input))
# device = torch.device("cuda: 3")
# model.to(device)
model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
# model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda(2)
# model = torch.nn.DataParallel(model)
# model.to(device)

# define loss function (criterion) and optimizer
# criterion = JointsMSELoss(
#     use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
# ).cuda(2)
criterion = JointsMSELoss(
    use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
).cuda()

# Data loading code
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


best_perf = 0.0
best_model = False
last_epoch = -1
optimizer = get_optimizer(cfg, model)
begin_epoch = cfg.TRAIN.BEGIN_EPOCH
checkpoint_file = os.path.join(
    final_output_dir, 'checkpoint.pth'
)

if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
    logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file)
    begin_epoch = checkpoint['epoch']
    best_perf = checkpoint['perf']
    last_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info("=> loaded checkpoint '{}' (epoch {})".format(
        checkpoint_file, checkpoint['epoch']))

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
    last_epoch=last_epoch
)

#%%
from my_utils import read_dlc_csv
csv_file = '/home/myp7435/Pop_freeReach_cam_2_0212-Min-2020-02-14/labeled-data/cam_2/CollectedData_Min.csv'
root_dir = '/home/myp7435/Pop_freeReach_cam_2_0212-Min-2020-02-14/'
df = read_dlc_csv(csv_file, root_dir)
#%%
from dataset.CustomDataset import CustomDataset

import numpy as np
idx = np.random.choice(len(df), int(len(df)*0.9), replace=False)
df_train = [df[i] for i in idx]
df_valid = [df[i] for i in np.arange(len(df)) if i not in idx]

train_dataset = CustomDataset(cfg, df_train, 
                              transform=transforms.Compose([transforms.ToTensor(),normalize,]))

valid_dataset = CustomDataset(cfg, df_valid, 
                              transform=transforms.Compose([transforms.ToTensor(),normalize,]))

#%%
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
    shuffle=cfg.TRAIN.SHUFFLE,
    num_workers=cfg.WORKERS,
    pin_memory=cfg.PIN_MEMORY
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
    shuffle=False,
    num_workers=cfg.WORKERS,
    pin_memory=cfg.PIN_MEMORY
)

#%%
for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
    lr_scheduler.step()

    # train for one epoch
    train(cfg, train_loader, model, criterion, optimizer, epoch,
          final_output_dir, tb_log_dir, writer_dict)
    best_model = True

#     evaluate on validation set
    perf_indicator = validate(
       cfg, valid_loader, valid_dataset, model, criterion,
       final_output_dir, tb_log_dir, writer_dict
   )

    if perf_indicator >= best_perf:
        best_perf = perf_indicator
        best_model = True
    else:
        best_model = False
    if perf_indicator <= best_perf:
        best_perf = perf_indicator
        best_model = True
    else:
        best_model = False

    logger.info('=> saving checkpoint to {}'.format(final_output_dir))
    save_checkpoint({
        'epoch': epoch + 1,
        'model': cfg.MODEL.NAME,
        'state_dict': model.state_dict(),
        'best_state_dict': model.module.state_dict(),
        'perf': perf_indicator,
        'optimizer': optimizer.state_dict(),
    }, best_model, final_output_dir)

final_model_state_file = os.path.join(
    final_output_dir, 'final_state.pth'
)
logger.info('=> saving final model state to {}'.format(
    final_model_state_file)
)
torch.save(model.module.state_dict(), final_model_state_file)
writer_dict['writer'].close()

#%%
from utils.my_utils import infer_video

vidpath = '/home/myp7435/Pop_freeReach_cam_2_0212-Min-2020-02-14/videos/cam_2_cropped4.mp4'
joints_name = ['Wrist', 'CMC_thumb', 'MCP_thumb', 'MCP1', 'MCP2', 'MCP3', 'MCP4',
          'IP_thumb', 'PIP1', 'PIP2', 'PIP3', 'PIP4', 'Dip1', 'Dip2', 'Dip3', 'Dip4',
          'Tip_thumb', 'Tip1', 'Tip2', 'Tip3', 'Tip4']
infer_video(vidpath, model, joints_name, transform=transforms.Compose([transforms.ToTensor(),normalize,]), downsample=8)

#%%
from utils.my_utils import create_labeled_video

vidpath = '/home/myp7435/Pop_freeReach_cam_2_0212-Min-2020-02-14/videos/cam_2_cropped4.mp4'
csvpath = '/home/myp7435/Pop_freeReach_cam_2_0212-Min-2020-02-14/videos/cam_2_cropped4_output.csv'
joints_name = ['Wrist', 'CMC_thumb', 'MCP_thumb', 'MCP1', 'MCP2', 'MCP3', 'MCP4',
          'IP_thumb', 'PIP1', 'PIP2', 'PIP3', 'PIP4', 'Dip1', 'Dip2', 'Dip3', 'Dip4',
          'Tip_thumb', 'Tip1', 'Tip2', 'Tip3', 'Tip4']
create_labeled_video(vidpath, csvpath, joints_name)
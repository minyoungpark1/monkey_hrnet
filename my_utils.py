#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:41:02 2019

@author: minyoungpark
"""

import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
import moviepy.editor as mp

import sys
sys.path.append('hrnet/lib')
from core.inference import get_max_preds
#
#csv_file = '/home/minyoungpark/Dropbox/Research/Miller/Project/Greyson_0919_cam4-Min-2019-10-07/labeled-data/cam4/CollectedData_Min.csv'
#root_dir = '/home/minyoungpark/Dropbox/Research/Miller/Project/Greyson_0919_cam4-Min-2019-10-07/'

def read_dlc_csv(csv_file, root_dir):
    
    df = pd.read_csv(csv_file)
    df.rename(columns=lambda x: df[x][0]+'_'+df[x][1], inplace=True)
    df = df.rename({"bodyparts_coords":"images"}, axis='columns')
    df = df.drop(df.index[0:2])
    df.index = range(len(df.index))
    num_joints = int((df.shape[1] - 1) / 2)
    db = []
    for idx in range(len(df)):
        image = os.path.join(root_dir, df.iloc[idx, 0])
        joints_3d = np.zeros((num_joints, 3))
        joints_3d_vis = np.zeros((num_joints, 3))
        
        joints_3d[:, :2] = df.iloc[idx, 1:].to_numpy().reshape((-1, 2))
        np.isfinite(joints_3d[:, :2], joints_3d_vis[:, :2])
        if np.isfinite(joints_3d[0, 0]):
            center = joints_3d[0, :2]
        else:
            img = cv2.imread(image)
            h, w, c = img.shape
            center = [w/2, h/2]
        joints_3d[np.isnan(joints_3d)] = 0
        db.append({'image': image,
                   'joints_3d': joints_3d,
                   'joints_3d_vis': joints_3d_vis,
                   'center': center,
                   'scale': 1
                   })
    
    return db


def read_dlc_labeled_data(root_dir, folders):
    data_paths = [os.path.join(root_dir, folder, 'CollectedData_Min.h5') \
              for folder in folders]
    db = []
    for data_path in data_paths:
        df = pd.read_hdf(data_path)
        images = [os.path.join(root_dir, os.path.join(*(image_path.split('/')[1:]))) \
          for image_path in df.index]
        num_joints = (df.shape[1])// 2
        for idx in range(len(df)):
            image = images[idx]
            joints_3d = np.zeros((num_joints, 3))
            joints_3d_vis = np.zeros((num_joints, 3))
            
            joints_3d[:, :2] = df.iloc[idx].to_numpy().reshape((-1, 2))
            np.isfinite(joints_3d[:, :2], joints_3d_vis[:, :2])
            if np.isfinite(joints_3d[0, 0]):
                center = joints_3d[0, :2]
            else:
                img = cv2.imread(image)
                h, w, c = img.shape
                center = [w/2, h/2]
            joints_3d[np.isnan(joints_3d)] = 0
            db.append({'image': image,
                       'joints_3d': joints_3d,
                       'joints_3d_vis': joints_3d_vis,
                       'center': center,
                       'scale': 1
                       })
    
    return db


def generate_downsampled_dataset(data_path, image_folders, save_path, scale):
    for image_folder in image_folders:
        save_folder = os.path.join(save_path, image_folder)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
            
        df = pd.read_hdf(os.path.join(data_path, image_folder, 'CollectedData_Min.h5'))
        df_down = df.copy()
        df_down = df * scale
        df_down.to_hdf(os.path.join(save_folder, 'CollectedData_Min.h5'), 'df_with_missing', mode='w')
        images = df_down.index
            
        for image in tqdm(images):
            image_name = os.path.basename(image)
            image_path = os.path.join(os.path.join(data_path, image_folder), image_name)
            img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            width = int(img.shape[1] * scale)
            height = int(img.shape[0] * scale)
            img_resize = cv2.resize(img, (width, height))
            cv2.imwrite(os.path.join(save_folder, image_name), img_resize)
            

def generate_downsampled_video(video_path, save_path, width):
    clip = mp.VideoFileClip(video_path)
    clip_resized = clip.resize(width=width) 
    clip_resized.write_videofile(save_path)
    cap = cv2.VideoCapture(save_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print((width, height))
    
    
def infer_video(vidpath, model, joints_name, transform, downsample=8):
    
    model.eval()
    
    cap = cv2.VideoCapture(vidpath)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    joints_col = np.repeat(joints_name, 3)
    sub = ['x', 'y', 'likelihood']
    sub_col = np.tile(sub, len(joints_name))

    df = pd.DataFrame(columns = [joints_col, sub_col])

    count = 0
    for idx in tqdm(range(frame_num)):
        ret, frame = cap.read()
        if ret == True:
            input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            if transform:
                input = transform(input)
            
            input = input.unsqueeze(dim=0)
            
            outputs = model(input)
            
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs
                
            preds, maxvals = get_max_preds(output.cpu().detach().numpy())
           
            joints = preds[0] * downsample
            joints_vis = maxvals[0]
            
            output_array = np.reshape(np.concatenate((joints, joints_vis), axis=1), (-1,))
            df.loc[count] = output_array
            count += 1
        else:
            break
            
    cap.release()
    
    csv_filename = vidpath.split('.')[0] + '_output.csv'
    df.to_csv(csv_filename)
    
    
def create_labeled_video(vidpath, csvpath, joints_name):
    cap = cv2.VideoCapture(vidpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_num/fps
    w = int(cap.get(3))
    h = int(cap.get(4))
    
#     colorclass = plt.cm.ScalarMappable(cmap='jet')
#     C = colorclass.to_rgba(np.linspace(0, 1, len(joints_name)))
#     colors = C[:, :3]
    
    labeled_vidpath = vidpath.split('.')[0] + '_output.mp4'
    out = cv2.VideoWriter(labeled_vidpath, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w, h))

    df = pd.read_csv(csvpath, header=[0, 1])
    
    for idx in tqdm(range(frame_num)):
        ret, frame = cap.read()
        if ret == True:
            for joint in joints_name:
                if df[joint,'likelihood'][idx] > 0.3:
                    frame = cv2.circle(frame, (int(df[joint,'x'][idx]), int(df[joint,'y'][idx])), 2, [255, 0, 0], 2)
            out.write(frame)
        else:
            break 
    cap.release()
    out.release()
    
    
def calc_avg_dist(output, target, threshold=0.5):
    cnt = 0
    preds, maxvals = get_max_preds(output)
    targets, _ = get_max_preds(target)
    
    preds = preds.astype(np.float32)
    targets = targets.astype(np.float32)
    maxvals = maxvals.astype(np.float32)
    
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            pred = preds[n, c, :]
            target = targets[n, c, :]
            if maxvals[n, c, 0] > threshold:
                dists[c, n] = np.linalg.norm(pred - target)
                cnt += 1
    if cnt > 0:
        avg_dist = np.sum(dists) / cnt
    else:
        avg_dist = 10000
        
    return avg_dist, cnt

def calc_avg_dist_v2(output, target, threshold=0.5):
    cnt = 0
    preds, maxvals = get_max_preds(output)
    targets, _ = get_max_preds(target)
    
    preds = preds.astype(np.float32)
    targets = targets.astype(np.float32)
    maxvals = maxvals.astype(np.float32)
    
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            pred = preds[n, c, :]
            target = targets[n, c, :]
            if maxvals[n, c, 0] > threshold:
                dists[c, n] = np.linalg.norm(pred - target)
                cnt += 1
    if cnt > 0:
        avg_dist = np.sum(dists) / cnt
    else:
        avg_dist = 10000
        
    return avg_dist, cnt
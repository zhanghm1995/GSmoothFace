'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-04-21 20:47:44
Email: haimingzhang@link.cuhk.edu.cn
Description: Prepare the dataset for training
'''

import cv2
import subprocess
import os
import os.path as osp
import face_alignment
import numpy as np
from tqdm import tqdm
from glob import glob


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)


def get_file_list(input_video, suffix="*.mp4"):
    """Get the video file path list

    Args:
        input_video (folder|single file path|files list): [description]

    Returns:
        list: list contains all video file absolute path
    """
    all_videos = []

    if type(input_video) == list:
        all_videos += input_video
    elif osp.isdir(input_video):
        all_videos = sorted(glob(osp.join(input_video, suffix)))
    else:
        all_videos += [input_video]
    return all_videos


def get_facial_landmarks(img_src):
    """Get facial landmarks array from input image

    Args:
        img_src (str|Path|cv2.Image): input image file path or image type

    Returns:
        np.ndarray: 68x2
    """
    if type(img_src) == "str":
        img_src = cv2.imread(img_src)
    
    try:
        preds = fa.get_landmarks_from_image(img_src)
    except:
        print(f"[Error] {img_src}")
    
    lm_pts = preds[0]
    return lm_pts


def get_five_landmark_pts(landmark_pts):
    """Get five positioning landmarks

    Args:
        landmark_pts (np.ndarray): (68, 3) or (68, 2)

    Returns:
        np.ndarray: (5, 3) or (5, 2)
    """
    lm_idx = np.array([31,37,40,43,46,49,55]) - 1
    ret = np.stack([landmark_pts[lm_idx[0],:], 
                    np.mean(landmark_pts[lm_idx[[1,2]],:],0), np.mean(landmark_pts[lm_idx[[3,4]],:],0), 
                    landmark_pts[lm_idx[5],:], landmark_pts[lm_idx[6],:]], axis = 0)
    ret = ret[[1,2,0,3,4], :]
    return ret


def main_extract_five_landmarks(image_file_list, save_all_landmarks=False):
    if not isinstance(image_file_list, list):
        image_file_list = get_file_list(image_file_list, suffix="*.jpg")
        print(f"There are {len(image_file_list)} images waiting for processing...")
    
    all_frames_landmarks = []
    
    prog_bar = tqdm(image_file_list)

    for img_file in prog_bar:
        prog_bar.set_description(f"Processing {img_file}")
        lm_pts = get_facial_landmarks(img_file)
        five_lm_pts = get_five_landmark_pts(lm_pts)

        ## Save five landmarks detection results into a txt file
        save_file = img_file.replace(".jpg", ".txt")
        np.savetxt(save_file, five_lm_pts, fmt="%.2f", delimiter=' ')

        all_frames_landmarks.append(lm_pts)
    
    if save_all_landmarks:
        ## Save all 68x2 landmarks results to a npy format file
        all_frames_landmarks_arr = np.stack(all_frames_landmarks)
        save_path = osp.join(osp.dirname(osp.dirname(img_file)), "landmarks.npy")
        np.save(save_path, all_frames_landmarks_arr)


if __name__ == "__main__":
    ## ---- Extract the landmarks
    main_extract_five_landmarks("./data/train_images/", save_all_landmarks=True)
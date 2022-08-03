'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-04-18 14:57:55
Email: haimingzhang@link.cuhk.edu.cn
Description: Create video by using images in a specific folder, and if given
             audio, could generate the video with audio signal
'''

import argparse
from typing import Tuple
import os
import os.path as osp
import cv2
import numpy as np
import shutil
from glob import glob
import subprocess
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='The arguments for creating video application')
    parser.add_argument('--image_root', type=str, default=None, required=True)
    parser.add_argument('--image_root2', type=str, default=None)
    parser.add_argument('--video_fps', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default="./demo")
    parser.add_argument('--audio_path', type=str, default=None)
    parser.add_argument('--video_name', type=str, default="video")
    # parser.add_argument('--keep_saving', action=True, help="true will not delete the output directory")

    args = parser.parse_args()
    return args


def create_video_writer(dst_file, dst_fps, dst_size: Tuple):
    """Create the OpenCV video writer

    Args:
        dst_file ([type]): [description]
        dst_fps ([type]): [description]
        dst_size (Tuple): please note it is (W,H) order

    Returns:
        (VideoWriter): [description]
    """
    if dst_file.endswith(".avi"):
        writer = cv2.VideoWriter(dst_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), dst_fps, dst_size)
        # writer = cv2.VideoWriter(dst_file, cv2.VideoWriter_fourcc(*"XVID"), dst_fps, dst_size)
    elif dst_file.endswith(".mp4"):
        writer = cv2.VideoWriter(dst_file, cv2.VideoWriter_fourcc(*"mp4v"), dst_fps, dst_size)
        # writer = cv2.VideoWriter(dst_file, cv2.VideoWriter_fourcc(*'XVID'), dst_fps, dst_size)
    else:
        raise ValueError("Unknown file format")

    return writer


def scan_image_folder(input_folder):
    return [file for file in os.listdir(input_folder) if file.endswith((".png", ".jpg"))]


def create_video_with_image_folder(image_root, 
                                   video_fps, 
                                   output_dir, 
                                   audio_path=None,
                                   video_name="video",
                                   need_avi_video=False):
    """Create a video file when given a folder contains all image files

    Args:
        image_root (str|Path): a folder path contains all image files
        video_fps (int): generated video FPS
        output_dir (str|Path): the destination folder
        video_name (str): the saving video file name
    """
    img_names = sorted(scan_image_folder(image_root))
    
    if not len(img_names):
        return
    
    ## Save the video
    output_video_name = f"{video_name}.avi"
    idx = -1
    for img_name in tqdm(img_names):
        idx += 1

        frame = cv2.imread(osp.join(image_root, img_name))
        
        if idx == 0:
            img_size = frame.shape[:2]
            writer = create_video_writer(osp.join(output_dir, output_video_name), 
                                         video_fps, 
                                         img_size[::-1])
        
        writer.write(frame)
    writer.release()

    ## Add audio channel into this video
    if audio_path is not None:
        ## We choose save .mp4 video by default

        video_path = osp.join(output_dir, output_video_name)

        if need_avi_video:
            ## Save .avi video
            output_video = osp.join(output_dir, f"{video_name}_with_audio.avi")
            command = f"ffmpeg -i {audio_path} -i {video_path} -vcodec copy  -acodec copy -y {output_video}"
            subprocess.call(command, shell=True, stdout=subprocess.DEVNULL)
        
        ## Save .mp4 video
        output_video = osp.join(output_dir, f"{video_name}_with_audio.mp4")
        command = f"ffmpeg -y -i {audio_path} -i {video_path} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {output_video}"
        subprocess.call(command, shell=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    args = parse_arguments()
    
    print("[WARNING] We will override the output directory!")
    if osp.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    create_video_with_image_folder(
        args.image_root, args.video_fps, args.output_dir, 
        audio_path=args.audio_path,
        video_name=args.video_name)

    print("Done")

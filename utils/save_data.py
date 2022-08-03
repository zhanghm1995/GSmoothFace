'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-20 23:44:25
Email: haimingzhang@link.cuhk.edu.cn
Description: Some useful functions to save data
'''
import os
import os.path as osp
import tempfile
import cv2
import subprocess
from tqdm import tqdm
import torch
from .utils import tensor2im
from scipy.io import wavfile


def save_video(image, output_video_fname, image_size=512, 
               audio_fname=None, audio_data=None, audio_sample_rate=16000):
    print("================== Start create the video =================================")
    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=osp.dirname(output_video_fname))
    writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 60, (image_size, image_size), True)
    for idx in tqdm(range(len(image))):
        writer.write(image[idx][..., ::-1])
    writer.release()

    print("================== Generate the final video with audio signal =====================")
    if audio_fname is not None:
        cmd = f'ffmpeg -y -i {audio_fname} -i {tmp_video_file.name} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {output_video_fname}'
    else:
        cmd = f'ffmpeg -y -i {tmp_video_file.name} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {output_video_fname}'
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Save video done!")


def save_image_array_to_video(image_array, output_dir, name=None, 
                              fps=25, rgb_mode=False, 
                              audio_array=None, audio_sample_rate=16000):
    """Save the image array into video

    Args:
        image_array (np.ndarray|Tensor): (B, T, 3, H, W)
        output_dir (_type_): _description_
        name (_type_, optional): _description_. Defaults to None.
        fps (int, optional): _description_. Defaults to 25.
        audio_array (np.ndarray|Tensor): (B, L). Defaults to None.
        audio_sample_rate (int, optional): _description_. Defaults to 16000.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    if torch.is_tensor(audio_array):
        audio_array = audio_array.cpu().numpy()

    for i in range(image_array.shape[0]):
        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=output_dir)
        for frame in range(image_array.shape[1]):
            image_numpy = tensor2im(image_array[i][frame])

            if rgb_mode:
                image_numpy = image_numpy[..., ::-1]
            
            if frame == 0:
                ## Create the video writer
                img_shape = image_numpy.shape[:2] # (H, W)
                writer = cv2.VideoWriter(tmp_video_file.name, 
                                         cv2.VideoWriter_fourcc(*'mp4v'), 
                                         fps, img_shape[::-1], True)
            writer.write(image_numpy)
        
        writer.release()

        if name is not None:
            output_video_fname = osp.join(output_dir, f"{name}_{i:03d}.mp4")
        else:
            output_video_fname = osp.join(output_dir, f"{i:03d}.mp4")

        ## Combine the audio
        if audio_array is not None:
            tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.wav', dir=output_dir)
            audio_data = audio_array[i]
            wavfile.write(tmp_audio_file.name, audio_sample_rate, audio_data)

            cmd = f'ffmpeg -y -i {tmp_audio_file.name} -i {tmp_video_file.name} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {output_video_fname}'
        else:
            cmd = f'ffmpeg -y -i {tmp_video_file.name} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {output_video_fname}'
        flag = subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def save_images(image_tensor, save_dir, epoch, global_step=None, name=None):
    """Save batched images

    Args:
        image_tensor (Tensor): (B, T, 3, H, W), T is the frames
        save_dir (str): save root directory
        epoch (int): epoch number
        global_step (_type_, optional): _description_. Defaults to None.
        name (_type_, optional): _description_. Defaults to None.
    """
    for i in range(image_tensor.shape[0]):
        for frame in range(image_tensor.shape[1]):
            image_numpy = tensor2im(image_tensor[i][frame])
            
            if name is not None:
                save_path = osp.join(save_dir, f"epoch_{epoch:03d}", f"{name}")
            else:
                save_path = osp.join(save_dir, f"epoch_{epoch:03d}", f"seq_{i:03d}")
            os.makedirs(save_path, exist_ok=True)

            cv2.imwrite(osp.join(save_path, f"{frame:06d}.jpg"), image_numpy)
            
        
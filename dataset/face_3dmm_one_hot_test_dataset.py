'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-27 14:39:56
Email: haimingzhang@link.cuhk.edu.cn
Description: Test dataset class to load arbitrary audio for testing
'''

import cv2
import os
import os.path as osp
from glob import glob
import torch
from torch import Tensor
from torch.utils.data import Dataset
import librosa
import numpy as np
from typing import List
from transformers import Wav2Vec2Processor
import torchvision.transforms as transforms
from .basic_bfm import BFMModel


def get_frames_from_video(video_path, num_need_frames=-1):
    """Read all frames from a video file

    Args:
        video_path (str): video file path

    Returns:
        list: including all images in OpenCV BGR format with HxWxC size
    """
    video_stream = cv2.VideoCapture(video_path)

    frames = []
    if num_need_frames < 0:
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            frames.append(frame)
    else:
        num_count = 0
        while num_count < num_need_frames:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            frames.append(frame)
            num_count += 1

    return frames


class Face3DMMTestDataset(Dataset):
    def __init__(
        self, 
        data_root,
        audio_path,
        video_name,
        training_split,
        fetch_length: int = 75,
        audio_sample_rate: int = 16000,
        video_fps: int = 25,
        **kwargs) -> None:
        super().__init__()

        self.data_root = data_root
        self.audio_path = audio_path
        self.video_name = video_name # current testing conditional speaker
        self.video_fps = video_fps

        training_split_names = open(osp.join(self.data_root, f'{training_split}.txt')).read().splitlines()

        self.fetch_length = fetch_length
        self.one_hot_labels = np.eye(len(training_split_names))
        self.one_hot_idx = training_split_names.index(self.video_name)

        self.audio_sample_rate = audio_sample_rate

        self.target_image_size = (192, 192)

        ## Define the image transformation operations
        transform_list = [transforms.ToTensor()]
        self.image_transforms = transforms.Compose(transform_list)

        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        self.facemodel = BFMModel("./data/BFM/BFM_model_front.mat")

        self._build_dataset()
        
    def _build_dataset(self):
        ## 1)  Read the audio data
        self.driven_audio_data, _ = librosa.load(self.audio_path, sr=self.audio_sample_rate)

        self.audio_stride = round(self.audio_sample_rate * self.fetch_length / self.video_fps)
        if self.audio_stride >= len(self.driven_audio_data):
            self.audio_chunks =[0]
        else:
            self.audio_chunks = range(0, len(self.driven_audio_data), self.audio_stride)

    def _read_image_sequence(self, image_path_list: List, need_mouth_masked_img: bool = False):
        img_list, mouth_masked_img_list = [], []
        for img_path in image_path_list:
            img = cv2.resize(cv2.imread(img_path), self.target_image_size)

            if need_mouth_masked_img:
                mouth_mask_img_path = img_path.replace("face_image", "mouth_mask").replace(".jpg", ".png")
                mouth_img = cv2.resize(cv2.imread(mouth_mask_img_path, cv2.IMREAD_UNCHANGED), self.target_image_size)
                mask2 = cv2.bitwise_not(mouth_img)
                mouth_masked_img = cv2.bitwise_and(img, img, mask=mask2)

                mouth_masked_img = self.image_transforms(mouth_masked_img) # To Tensor
                mouth_masked_img_list.append(mouth_masked_img)

            img = self.image_transforms(img)
            img_list.append(img)

        img_seq_tensor = torch.stack(img_list) # to (T, 3, H, W)

        if need_mouth_masked_img:
            mouth_masked_img_tensor =  torch.stack(mouth_masked_img_list)
            return img_seq_tensor, mouth_masked_img_tensor
        else:
            return img_seq_tensor

    def __len__(self):
        return len(self.audio_chunks)
    
    def _get_template(self, choose_video):
        ## Assume the first frame is the template face
        video_path = osp.join(self.data_root, choose_video)

        template_face = np.load(osp.join(video_path, "template_face.npy"))
        id_coeff = np.load(osp.join(video_path, "id_coeff.npy"))
        return template_face, id_coeff

    def __getitem__(self, index):
        one_hot = self.one_hot_labels[self.one_hot_idx]

        ## 1) Read the audio
        audio_start_idx = self.audio_chunks[index]

        audio_seq = self.driven_audio_data[audio_start_idx: audio_start_idx + self.audio_stride]

        ## Extract the audio features
        audio_seq = np.squeeze(self.audio_processor(audio_seq, sampling_rate=self.audio_sample_rate).input_values)

        actual_frame_lenth = int(len(audio_seq) / self.audio_sample_rate * self.video_fps)

        ## Get the template info
        template_face, id_coeff = self._get_template(self.video_name)

        data_dict = {}

        data_dict['raw_audio'] = torch.tensor(audio_seq.astype(np.float32))
        data_dict['one_hot'] = torch.FloatTensor(one_hot)
        data_dict['template'] = torch.FloatTensor(template_face.reshape((-1))) # (N,)
        data_dict['video_name']  = self.video_name
        data_dict['exp_base'] = torch.FloatTensor(self.facemodel.exp_base)
        return data_dict

'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-20 20:45:23
Email: haimingzhang@link.cuhk.edu.cn
Description: Base dataset class for loading video dataset sequence
'''

import os
import os.path as osp
from typing import List
import numpy as np
import random
from glob import glob
import cv2
from torch.utils.data import Dataset
import librosa
import torchvision.transforms as transforms
import torch

def get_all_valid_indices(total_length, fetch_length, stride) -> List:
    idx_list = list(range(0, total_length - fetch_length, stride))
    # last_idx = total_length - fetch_length
    # idx_list += [last_idx]
    return idx_list


class BaseVideoDataset(Dataset):
    """Class to load a list of images and corresponding audio data.
       Currently just load one person all videos (means there are multiple
       image sets belongs to single person waiting for load)

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, split, **kwargs) -> None:
        super().__init__()
        

        self.data_root = kwargs['data_root']
        
        flag = osp.exists(self.data_root)
        assert flag, f"{self.data_root} is not exist, please check!"

        self.all_videos_dir = open(osp.join(self.data_root, f'{split}.txt')).read().splitlines()

        self.fetch_length = kwargs.get("fetch_length", 75)
        self.fetch_stride = kwargs.get("fetch_stride", 50)
        self.video_fps = kwargs.get("video_fps", 25)
        self.audio_sample_rate = kwargs.get("audio_sample_rate", 16000)
        self.target_image_size = kwargs.get("target_image_size", 192)

        self.build_dataset()

        ## Define the image transformation operations
        transform_list = [transforms.ToTensor()]
        self.image_transforms = transforms.Compose(transform_list)

    def build_dataset(self):
        self.total_frames_list = []
        self.length_token_list = [] # increamental length list
        self.all_sliced_indices = [] # list of list

        total_length = 0
        for video_dir in self.all_videos_dir:
            # all_images_path = sorted(glob(osp.join(self.data_root, video_dir, "face_image", "*.jpg")))
            image_dir = osp.join(self.data_root, video_dir, "face_image")
            all_images_path = sorted([file.path for file in os.scandir(image_dir) if file.name.endswith(".jpg")])
            num_frames = len(all_images_path)
            self.total_frames_list.append(num_frames)

            valid_indices = get_all_valid_indices(num_frames, self.fetch_length, stride=self.fetch_stride)
            self.all_sliced_indices.append(valid_indices)

            total_length += len(valid_indices)
            self.length_token_list.append(total_length)
    
    def _get_data(self, index):
        """Get the seperate index location from the total index

        Args:
            index (int): index in all avaible sequeneces
        
        Returns:
            main_idx (int): index specifying which video
            sub_idx (int): index specifying what the start index in this sliced video
        """
        def fetch_data(length_list, index):
            assert index < length_list[-1]
            temp_idx = np.array(length_list) > index
            list_idx = np.where(temp_idx==True)[0][0]
            sub_idx = index
            if list_idx != 0:
                sub_idx = index - length_list[list_idx - 1]
            return list_idx, sub_idx

        main_idx, sub_idx = fetch_data(self.length_token_list, index)
        return main_idx, sub_idx

    def __len__(self):
        return sum([len(x) for x in self.all_sliced_indices])
    
    def _slice_raw_audio(self, choose_video, start_index):
        """Slice the raw whole audio into a vector with fetch length

        Args:
            choose_video (str): choosed video directory path
            start_index (int): choosed the video start index

        Returns:
            np.ndarray: (M, )
        """
        audio_path = osp.join(self.data_root, choose_video, f"{osp.basename(choose_video)}.wav")
        
        start_idx, end_idx = start_index, start_index + self.fetch_length
        audio_start_idx = round(start_idx / self.video_fps * self.audio_sample_rate)
        audio_end_idx = round(end_idx / self.video_fps * self.audio_sample_rate)

        audio_idx_diff = audio_end_idx - audio_start_idx
        
        whole_audio_data, _ = librosa.load(audio_path, sr=self.audio_sample_rate)
        fetch_audio_data = whole_audio_data[audio_start_idx:audio_end_idx]

        if len(fetch_audio_data) != audio_idx_diff:
            return None
        
        return fetch_audio_data

    def _get_reference_image(self, video_length, video_dir, choose_idx):
        def get_valid_list_indices(list_length, choose_idx, forbidden_length):
            origin_list = list(range(list_length))
            forbidden_start_idx = max(0, (choose_idx - int(forbidden_length / 2)))
            forbidden_end_idx = min(list_length, (choose_idx + int(forbidden_length / 2)))
            sub_list = list(set(origin_list) - set(origin_list[forbidden_start_idx:forbidden_end_idx]) - 
                            set(origin_list[-forbidden_length:]))
            return sub_list
        
        valid_indices = get_valid_list_indices(video_length, choose_idx, self.fetch_length) # list
        
        ### randomly choose one start index
        ref_start_idx = random.sample(valid_indices, k=1)[0]
        
        return ref_start_idx

    def _read_image_sequence(self, video_dir, start_idx, need_mouth_masked_img=False):
        img_list, mouth_masked_img_list = [], []
        
        for idx in range(start_idx, start_idx + self.fetch_length):
            ## Read the face image and resize
            img_path = osp.join(self.data_root, video_dir, "face_image", f"{idx:06d}.jpg")
            img = cv2.resize(cv2.imread(img_path), self.target_image_size)

            if need_mouth_masked_img:
                img_path = osp.join(self.data_root, video_dir, "mouth_mask", f"{idx:06d}.png")
                mouth_mask_img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED),
                                            self.target_image_size, 
                                            interpolation=cv2.INTER_NEAREST)[..., None] # (H, W, 1)

                mouth_mask_img = self.image_transforms(mouth_mask_img)
                mouth_masked_img_list.append(mouth_mask_img)
            
            img = self.image_transforms(img)
            img_list.append(img)
        
        img_seq_tensor = torch.stack(img_list) # to (T, 3, H, W)

        if need_mouth_masked_img:
            mouth_masked_img_tensor =  torch.stack(mouth_masked_img_list)
            return img_seq_tensor, mouth_masked_img_tensor
        else:
            return img_seq_tensor

    def __getitem__(self, index):
        main_idx, sub_idx = self._get_data(index)
        
        choose_video = self.all_videos_dir[main_idx] # choosed video directory name, str type
        start_idx = self.all_sliced_indices[main_idx][sub_idx]

        audio_seq = self._slice_raw_audio(choose_video, sub_idx)
        if audio_seq is None:
            return None

        ## Get the GT image
        gt_img_seq_tensor = self._read_image_sequence(choose_video, start_idx)

        ## Get the reference image
        ref_img_seq_tensor, _ = self._get_reference_image(
            self.total_frames_list[main_idx], choose_video, start_idx)
        
        data_dict = {}
        data_dict['gt_face_image'] = gt_img_seq_tensor
        data_dict['ref_face_image'] = ref_img_seq_tensor
        data_dict['raw_audio'] = torch.tensor(audio_seq.astype(np.float32))
        return data_dict


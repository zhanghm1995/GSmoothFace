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
from .basic_bfm import BFMModelPyTorch
from .face_3dmm_utils import read_face3dmm_params
from .dataset_utils import load_split_file


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
        self.audio_sample_rate = audio_sample_rate

        self.use_template_face = False

        training_split_names = load_split_file(self.data_root, training_split)

        self.fetch_length = fetch_length
        self.one_hot_labels = np.eye(len(training_split_names))
        self.one_hot_idx = training_split_names.index(self.video_name)

        ## Get the video GT face parameters for visualization
        curr_deep3dface_dir = osp.join(self.data_root, self.video_name, "deep3dface")
        self.face_3dmm_mat_list = sorted(glob(osp.join(curr_deep3dface_dir, "*.mat")))

        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        self.facemodel = BFMModelPyTorch("./data/BFM/BFM_model_front.mat")

        self._build_dataset()

    def _build_dataset(self):
        ## 1)  Read the audio data
        self.driven_audio_data, _ = librosa.load(self.audio_path, sr=self.audio_sample_rate)

        self.audio_stride = round(self.audio_sample_rate * self.fetch_length / self.video_fps)
        if self.audio_stride >= len(self.driven_audio_data):
            self.audio_chunks =[0]
        else:
            self.audio_chunks = range(0, len(self.driven_audio_data), self.audio_stride)

    def __len__(self):
        return len(self.audio_chunks)
    
    def _get_template(self, choose_video):
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

        data_dict = {}
        ## Get the template info
        if self.use_template_face:
            template_face, id_coeff = self._get_template(self.video_name)
            data_dict['template'] = torch.FloatTensor(template_face.reshape((-1))) # (N,)
        else:
            ## TODO: Currently we only use the first frame ID parameters to compute the template vertex
            mat_file = self.face_3dmm_mat_list[0]
            
            face_3dmm_params = read_face3dmm_params(mat_file) # (1, 257)
            origin_id_coeffs = torch.from_numpy(face_3dmm_params[:, :80])

            template_face = self.facemodel.compute_shape(
                id_coeff=origin_id_coeffs, exp_coeff=None) # (T, 3N)
            data_dict['template'] = template_face

        data_dict['raw_audio'] = torch.tensor(audio_seq.astype(np.float32))
        data_dict['one_hot'] = torch.FloatTensor(one_hot)
        data_dict['video_name']  = self.video_name
        data_dict['exp_base'] = torch.FloatTensor(self.facemodel.exp_base)
        return data_dict

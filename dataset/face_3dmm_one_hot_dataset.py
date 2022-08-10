'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-22 10:17:20
Email: haimingzhang@link.cuhk.edu.cn
Description: The dataset to get one-hot vector like official FaceFormer
'''


import os.path as osp
import numpy as np
import torch
from transformers import Wav2Vec2Processor
import time
from PIL import Image

import torchvision.transforms as transforms

from .base_video_dataset import BaseVideoDataset
from .basic_bfm import BFMModel, BFMModelPyTorch
from .face_3dmm_utils import get_face_3d_params
from .dataset_utils import read_image_sequence


class Face3DMMOneHotDataset(BaseVideoDataset):
    def __init__(
        self, 
        data_root, 
        split, 
        use_template_face: bool = False,
        need_load_image: bool = False,
        **kwargs) -> None:
        super(Face3DMMOneHotDataset, self).__init__(data_root, split, **kwargs)

        self.use_template_face = use_template_face
        self.need_load_image = need_load_image

        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        self.one_hot_labels = np.eye(len(self.all_videos_dir))
        # self.one_hot_labels = np.eye(8)

        self.facemodel = BFMModelPyTorch("./data/BFM/BFM_model_front.mat")

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])

    def _get_template(self, choose_video):
        ## Assume the first frame is the template face
        video_path = osp.join(self.data_root, choose_video)

        template_face = np.load(osp.join(video_path, "template_face.npy"))
        id_coeff = np.load(osp.join(video_path, "id_coeff.npy"))
        return template_face, id_coeff
        
    def __getitem__(self, index):
        main_idx, sub_idx = self._get_data(index)

        one_hot = self.one_hot_labels[main_idx]
        
        choose_video = self.all_videos_dir[main_idx] # choosed video directory name, str type
        start_idx = self.all_sliced_indices[main_idx][sub_idx] # the actual index in this video

        data_dict = {}

        ## Get the GT raw audio vector
        audio_seq = self._slice_raw_audio(choose_video, start_idx) # (M, )
        if audio_seq is None:
            return None
        
        audio_seq = np.squeeze(self.audio_processor(audio_seq, sampling_rate=self.audio_sample_rate).input_values)
        data_dict['raw_audio'] = torch.tensor(audio_seq.astype(np.float32)) #(L, )

        ## Get the GT 3D face parameters
        # face_3d_params_dict = self._get_face_3d_params(choose_video, start_idx, need_origin_params=True)
        face_3d_params_dict = get_face_3d_params(self.data_root, choose_video, start_idx, 
                                                 need_origin_params=True, 
                                                 fetch_length=self.fetch_length)
        data_dict.update(face_3d_params_dict)

        gt_face_3d_params_arr = face_3d_params_dict['gt_face_3d_params']

        if self.use_template_face:
            ## Get the template info
            template_face, id_coeff = self._get_template(choose_video)

            ## Get the GT 3D face vertex (T, 3N)
            gt_face_3d_vertex = self.facemodel.compute_shape(
                id_coeff=id_coeff, exp_coeff=gt_face_3d_params_arr)
            
            data_dict['template'] = torch.FloatTensor(template_face.reshape((-1))) # (N,)
            data_dict['face_vertex'] = torch.FloatTensor(gt_face_3d_vertex)
        else:
            origin_id_coeffs = face_3d_params_dict['gt_face_origin_3d_params'][:, :80]
            template_face = self.facemodel.compute_shape(
                id_coeff=origin_id_coeffs, exp_coeff=None) # (T, 3N)

            gt_face_3d_vertex = self.facemodel.compute_shape(
                id_coeff=origin_id_coeffs, exp_coeff=gt_face_3d_params_arr)
            
            data_dict['template'] = template_face
            data_dict['face_vertex'] = gt_face_3d_vertex
        
        if self.need_load_image:
            gt_img_seq_tensor = read_image_sequence(
                self.data_root, choose_video, start_idx, self.fetch_length, self.transform)
            data_dict['gt_face_image_seq'] = gt_img_seq_tensor
        
        data_dict['one_hot'] = torch.FloatTensor(one_hot)
        data_dict['video_name']  = choose_video
        data_dict['exp_base'] = torch.FloatTensor(self.facemodel.exp_base)
        return data_dict

'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-22 10:17:20
Email: haimingzhang@link.cuhk.edu.cn
Description: The class to load 2D and 3D face dataset
'''


import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from transformers import Wav2Vec2Processor
from .base_video_dataset import BaseVideoDataset
from .face_3dmm_one_hot_dataset import Face3DMMOneHotDataset


class Face2D3DDataset(Face3DMMOneHotDataset):
    def __init__(self, split, **kwargs) -> None:
        super(Face2D3DDataset, self).__init__(split, **kwargs)
        self.need_origin_face_3d_param = kwargs.get("need_origin_face_3d_param", False)

    def __getitem__(self, index):
        main_idx, sub_idx = self._get_data(index)

        one_hot = self.one_hot_labels[main_idx]
        
        choose_video = self.all_videos_dir[main_idx] # choosed video directory name, str type
        start_idx = self.all_sliced_indices[main_idx][sub_idx] # the actual index in this video

        ## Get the GT raw audio vector
        audio_seq = self._slice_raw_audio(choose_video, start_idx) # (M, )
        if audio_seq is None:
            return None
        
        audio_seq = np.squeeze(self.audio_processor(audio_seq, sampling_rate=16000).input_values)

        ## Get the GT 3D face parameters
        gt_face_3d_params_arr, gt_face_origin_3d_param = self._get_face_3d_params(
            choose_video, start_idx, need_origin_params=self.need_origin_face_3d_param)

        ## Get the template info
        template_face, id_coeff = self._get_template(choose_video)

        ## Get the GT 3D face vertex ()
        gt_face_3d_vertex = self.facemodel.compute_shape(
            id_coeff=id_coeff, exp_coeff=gt_face_3d_params_arr)
        
        ## Get the 2D face image
        gt_img_seq_tensor, gt_img_mouth_mask_tensor = self._read_image_sequence(
            choose_video, start_idx, need_mouth_masked_img=True)

        data_dict = {}
        data_dict['gt_face_3d_params'] = torch.from_numpy(gt_face_3d_params_arr.astype(np.float32)) # (fetch_length, 64)
        data_dict['raw_audio'] = torch.tensor(audio_seq.astype(np.float32)) #(L, )
        data_dict['one_hot'] = torch.FloatTensor(one_hot)
        data_dict['template'] = torch.FloatTensor(template_face.reshape((-1))) # (N,)
        data_dict['face_vertex'] = torch.FloatTensor(gt_face_3d_vertex)
        data_dict['video_name']  = choose_video
        data_dict['gt_face_image'] = gt_img_seq_tensor # (fetch_length, 3, H, W)
        data_dict['gt_mouth_mask_image'] = gt_img_mouth_mask_tensor # (fetch_length, 1, H, W)

        if self.need_origin_face_3d_param:
            # (fetch_length, 257)
            data_dict['gt_face_origin_3d_params'] = torch.from_numpy(gt_face_origin_3d_param.astype(np.float32))

        return data_dict

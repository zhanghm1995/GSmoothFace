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
import cv2
from PIL import Image
from torch.utils.data import Dataset
from scipy.io import loadmat
from transformers import Wav2Vec2Processor
from .base_video_dataset import BaseVideoDataset
from .face_2d_3d_dataset import Face2D3DDataset
from utils.preprocess import align_img


# load landmarks for standard face, which is used for image preprocessing
def load_lm3d(bfm_folder="./data/BFM"):

    Lm3D = loadmat(osp.join(bfm_folder, 'similarity_Lm3D_all.mat'))
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D


class FaceDeep3DDataset(Face2D3DDataset):
    def __init__(self, split, **kwargs) -> None:
        super(FaceDeep3DDataset, self).__init__(split, **kwargs)
        self.need_preprocess_face_image = kwargs.get("need_preprocess_face_image", False)
        self.lm3d_std = load_lm3d()

    def read_image_sequence(self, video_dir, start_idx, need_mouth_masked_img=False):
        """Read image sequence like Deeep3DFace_Pytorch does

        Args:
            video_dir (str): _description_
            start_idx (str): _description_
            need_mouth_masked_img (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        img_list, mouth_masked_img_list = [], []
        face_mask_img_list = []
        
        for idx in range(start_idx, start_idx + self.fetch_length):
            ## Read the face image and resize
            video_dir_path = osp.join(self.data_root, video_dir)
            img_path = osp.join(video_dir_path, "face_image", f"{idx:06d}.jpg")
            lm_path = osp.join(video_dir_path, "face_image", f"{idx:06d}.txt")
            mouth_mask_path = osp.join(video_dir_path, "mouth_mask", f"{idx:06d}.png")
            face_mask_path = osp.join(video_dir_path, "face_mask_224", f"{idx:06d}.png")

            raw_img = Image.open(img_path).convert('RGB')
            raw_mouth_msk = Image.open(mouth_mask_path).convert('RGB') if need_mouth_masked_img else None
            raw_lm = np.loadtxt(lm_path).astype(np.float32)

            W, H = raw_img.size
            raw_lm = raw_lm.reshape([-1, 2])
            raw_lm[:, -1] = H - 1 - raw_lm[:, -1]
            _, img, lm, msk, _ = align_img(raw_img, raw_lm, self.lm3d_std, raw_mouth_msk)

            face_mask_img = cv2.imread(face_mask_path, cv2.IMREAD_UNCHANGED)[..., None] # (H, W, 1)

            if need_mouth_masked_img:
                mouth_mask_img = self.image_transforms(msk)[:1, ...]
                mouth_masked_img_list.append(mouth_mask_img)
            
            img = self.image_transforms(img)
            img_list.append(img)

            face_mask_img = self.image_transforms(face_mask_img)
            face_mask_img_list.append(face_mask_img)
        
        img_seq_tensor = torch.stack(img_list) # to (T, 3, H, W)
        face_mask_img_tensor = torch.stack(face_mask_img_list) # to (T, 1, H, W)
        
        if need_mouth_masked_img:
            mouth_masked_img_tensor =  torch.stack(mouth_masked_img_list)
            return {"gt_face_img": img_seq_tensor,
                    'face_mask_img': face_mask_img_tensor,
                    'mouth_mask_img': mouth_masked_img_tensor}
        else:
            return {'gt_face_img': img_seq_tensor,
                    'face_mask_img': face_mask_img_tensor}

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
            id_coeff=gt_face_origin_3d_param[:, :80], exp_coeff=gt_face_3d_params_arr)
        
        ## Get the 2D face image
        img_tensor_dict = self.read_image_sequence(
            choose_video, start_idx, need_mouth_masked_img=True)

        data_dict = {}
        data_dict['gt_face_3d_params'] = torch.from_numpy(gt_face_3d_params_arr.astype(np.float32)) # (fetch_length, 64)
        data_dict['raw_audio'] = torch.tensor(audio_seq.astype(np.float32)) #(L, )
        data_dict['one_hot'] = torch.FloatTensor(one_hot)
        data_dict['template'] = torch.FloatTensor(template_face.reshape((-1))) # (N,)
        data_dict['face_vertex'] = torch.FloatTensor(gt_face_3d_vertex)
        data_dict['video_name']  = choose_video
        data_dict['gt_face_image'] = img_tensor_dict['gt_face_img'] # (fetch_length, 3, H, W)
        data_dict['gt_mouth_mask_image'] = img_tensor_dict['mouth_mask_img'] # (fetch_length, 1, H, W)
        data_dict['gt_face_mask_image'] = img_tensor_dict['face_mask_img'] # (fetch_length, 1, H, W)
        data_dict['gt_masked_face_image'] = data_dict['gt_face_image'] * data_dict['gt_face_mask_image']
        data_dict['exp_base'] = torch.FloatTensor(self.facemodel.exp_base)
        
        if self.need_origin_face_3d_param:
            # (fetch_length, 257)
            data_dict['gt_face_origin_3d_params'] = \
                torch.from_numpy(gt_face_origin_3d_param.astype(np.float32))

        return data_dict

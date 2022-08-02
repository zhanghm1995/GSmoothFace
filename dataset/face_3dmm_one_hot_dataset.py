'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-22 10:17:20
Email: haimingzhang@link.cuhk.edu.cn
Description: The dataset to get one-hot vector like official FaceFormer
'''


import os.path as osp
import numpy as np
import scipy.io as spio
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from transformers import Wav2Vec2Processor
from .base_video_dataset import BaseVideoDataset
from .basic_bfm import BFMModel
from .face_3dmm_utils import get_face_3d_params

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def loadmat2(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


class Face3DMMOneHotDataset(BaseVideoDataset):
    def __init__(self, data_root, split, **kwargs) -> None:
        super(Face3DMMOneHotDataset, self).__init__(data_root, split, **kwargs)
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        self.one_hot_labels = np.eye(len(self.all_videos_dir))
        # self.one_hot_labels = np.eye(8)

        self.facemodel = BFMModel("./data/BFM/BFM_model_front.mat")
        
    def _get_mat_vector(self, face_params_dict,
                        keys_list=['id', 'exp', 'tex', 'angle', 'gamma', 'trans']):
        """Get coefficient vector from Deep3DFace_Pytorch results

        Args:
            face_params_dict (dict): face params dictionary loaded by using loadmat function

        Returns:
            np.ndarray: (1, L)
        """

        coeff_list = []
        for key in keys_list:
            coeff_list.append(face_params_dict[key])
        
        coeff_res = np.concatenate(coeff_list, axis=1)
        return coeff_res

    def _get_face_3d_params(self, video_dir, start_idx, need_origin_params=False):
        """Get face 3d params from a video and specified start index

        Args:
            video_dir (str): video name
            start_idx (int): start index

        Returns:
            np.ndarray: (L, C), L is the fetch length, C is the needed face parameters dimension
        """
        face_3d_params_list, face_origin_3d_params_list = [], []
        trans_matrix_list = []
        for idx in range(start_idx, start_idx + self.fetch_length):
            face_3d_params_path = osp.join(self.data_root, video_dir, "deep3dface", f"{idx:06d}.mat")
            
            face_3d_params_dict = loadmat(face_3d_params_path) # dict type

            if need_origin_params:
                face_origin_3d_params = self._get_mat_vector(face_3d_params_dict) # (1, 257)
                face_3d_params = face_origin_3d_params[:, 80:144]

                face_origin_3d_params_list.append(face_origin_3d_params)
            else:
                face_3d_params = self._get_mat_vector(face_3d_params_dict, keys_list=["exp"])
            
            face_3d_params_list.append(face_3d_params)

            trans_matrix_list.append(loadmat2(face_3d_params_path)['transform_params'])

        res_dict = dict()
        
        res_dict['gt_face_3d_params'] = np.concatenate(face_3d_params_list, axis=0) # (T, 64)
        # res_dict['trans_matrix'] = torch.FloatTensor(np.concatenate(trans_matrix_list, axis=0))
        
        if need_origin_params:
            # (T, 257)
            res_dict['gt_face_origin_3d_params'] = torch.FloatTensor(np.concatenate(face_origin_3d_params_list, axis=0))

        return res_dict

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
        face_3d_params_dict = get_face_3d_params(self.data_root, choose_video, start_idx, need_origin_params=True, 
                                                 fetch_length=self.fetch_length)
        data_dict.update(face_3d_params_dict)

        gt_face_3d_params_arr = face_3d_params_dict['gt_face_3d_params']

        ## Get the template info
        template_face, id_coeff = self._get_template(choose_video)

        ## Get the GT 3D face vertex ()
        gt_face_3d_vertex = self.facemodel.compute_shape(
            id_coeff=id_coeff, exp_coeff=gt_face_3d_params_arr)
        
        data_dict['gt_face_3d_params'] = torch.from_numpy(gt_face_3d_params_arr.astype(np.float32)) # (fetch_length, 64)
        data_dict['one_hot'] = torch.FloatTensor(one_hot)
        data_dict['template'] = torch.FloatTensor(template_face.reshape((-1))) # (N,)
        data_dict['face_vertex'] = torch.FloatTensor(gt_face_3d_vertex)
        data_dict['video_name']  = choose_video
        data_dict['exp_base'] = torch.FloatTensor(self.facemodel.exp_base)
        return data_dict



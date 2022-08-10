'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-07-31 10:47:40
Email: haimingzhang@link.cuhk.edu.cn
Description: The utility functions to load the 3DMM face parameters
and apply some processings.
'''

import os
import os.path as osp
from scipy.io import loadmat
import scipy.io as spio
import torch
import numpy as np


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


def get_coeff_vector(face_params_dict, key_list=None, reset_list=None):
    """Get coefficient vector from Deep3DFace_Pytorch results

    Args:
        face_params_dict (dict): the dictionary contains reconstructed 3D face

    Returns:
        [np.ndarray]: 1x257
    """
    if key_list is None:
        keys_list = ['id', 'exp', 'tex', 'angle', 'gamma', 'trans']
    else:
        keys_list = key_list

    coeff_list = []
    for key in keys_list:
        if reset_list is not None and key in reset_list:
            value = np.zeros_like(face_params_dict[key])
            coeff_list.append(value)
        else:
            coeff_list.append(face_params_dict[key])
    
    coeff_res = np.concatenate(coeff_list, axis=1)
    return coeff_res


def get_face_3d_params(
    data_root,
    video_dir, 
    start_idx, 
    need_origin_params=False,
    fetch_length: int = 100):
    """Get face 3d params from a video and specified start index

    Args:
        video_dir (str): video name
        start_idx (int): start index

    Returns:
        np.ndarray: (L, C), L is the fetch length, C is the needed face parameters dimension
    """
    face_3d_params_list, face_origin_3d_params_list = [], []
    trans_matrix_list = []
    for idx in range(start_idx, start_idx + fetch_length):
        face_3d_params_path = osp.join(data_root, video_dir, "deep3dface", f"{idx:06d}.mat")
        
        face_3d_params_dict = loadmat(face_3d_params_path) # dict type

        if need_origin_params:
            face_origin_3d_params = get_coeff_vector(face_3d_params_dict) # (1, 257)
            face_3d_params = face_origin_3d_params[:, 80:144]

            face_origin_3d_params_list.append(face_origin_3d_params)
        else:
            face_3d_params = get_coeff_vector(face_3d_params_dict, keys_list=["exp"])
        
        face_3d_params_list.append(face_3d_params)

        # trans_matrix_list.append(loadmat2(face_3d_params_path)['transform_params'])

    res_dict = dict()
    
    res_dict['gt_face_3d_params'] = torch.FloatTensor(np.concatenate(face_3d_params_list, axis=0)) # (T, 64)
    # res_dict['trans_matrix'] = torch.FloatTensor(np.concatenate(trans_matrix_list, axis=0))
    
    if need_origin_params:
        # (T, 257)
        res_dict['gt_face_origin_3d_params'] = torch.FloatTensor(np.concatenate(face_origin_3d_params_list, axis=0))

    return res_dict


def get_face_exp_params_sequence(
    data_root,
    video_dir, 
    start_idx, 
    fetch_length: int = 100,
    need_origin_params=False,
    need_crop_params: bool = False):
    """Get face 3d params from a video and specified start index

    Args:
        video_dir (str): video name
        start_idx (int): start index

    Returns:
        np.ndarray: (L, C), L is the fetch length, C is the needed face parameters dimension
    """
    face_3d_params_list, face_origin_3d_params_list = [], []
    for idx in range(start_idx, start_idx + fetch_length):
        face_3d_params_path = osp.join(data_root, video_dir, "deep3dface", f"{idx:06d}.mat")
        face_3mm_params_all = read_face3dmm_params(face_3d_params_path, need_crop_params=need_crop_params) # (1, 260)
        
        face_exp_params = face_3mm_params_all[:, 80:144] # (1, 64)

        if need_origin_params:
            face_origin_3d_params_list.append(face_3mm_params_all)
        
        face_3d_params_list.append(face_exp_params)

    res_dict = dict()
    
    res_dict['gt_face_3d_params'] = torch.FloatTensor(np.concatenate(face_3d_params_list, axis=0)) # (T, 64)
    
    if need_origin_params:
        # (T, 257)
        res_dict['gt_face_origin_3d_params'] = torch.FloatTensor(np.concatenate(face_origin_3d_params_list, axis=0))

    return res_dict


def read_face3dmm_params(file_path, need_crop_params=False):
    assert file_path.endswith(".mat")

    file_mat = loadmat(file_path)
    coeff_3dmm = get_coeff_vector(file_mat)
    
    if need_crop_params:
        crop_param = file_mat['transform_params']
        _, _, ratio, t0, t1 = np.hsplit(crop_param.astype(np.float32), 5)
        crop_param = np.concatenate([ratio, t0, t1], 1)
        coeff_3dmm = np.concatenate([coeff_3dmm, crop_param], axis=1)

    return coeff_3dmm
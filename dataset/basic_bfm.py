'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-27 21:56:15
Email: haimingzhang@link.cuhk.edu.cn
Description: The basic BFM model
'''

import numpy as np
from scipy.io import loadmat
import time
import torch


class BFMModel(object):
    def __init__(self, BFM_model_path="BFM/BFM_model_front.mat", 
                 recenter=True) -> None:
        model = loadmat(BFM_model_path)
        self.mean_shape = model['meanshape'].astype(np.float32) # [3*N,1]
        self.id_base = model['idBase'].astype(np.float32) # [3*N,80]
        self.exp_base = model['exBase'].astype(np.float32) # [3*N,64]
        self.key_point = np.squeeze(model['keypoints']).astype(np.int64) - 1
        if recenter:
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape = mean_shape - np.mean(mean_shape, axis=0, keepdims=True)
            self.mean_shape = mean_shape.reshape([-1, 1])

    def compute_shape(self, id_coeff=None, exp_coeff=None):
        """Compute the complete 3D face shape

        Args:
            id_coeff (np.ndarray): (1, 80)
            exp_coeff (np.ndarray): (1, 64)

        Returns:
            np.ndarray: (B, 3*N)
        """
        if id_coeff is None:
            id_coeff = np.zeros((1, 80)).astype(self.id_base.dtype)
        if exp_coeff is None:
            exp_coeff = np.zeros((1, 64)).astype(self.exp_base.dtype)

        start = time.time()
        id_info = id_coeff @ self.id_base.T
        end = time.time()
        print("Computing shape takes {} seconds".format(end - start), self.id_base.shape, id_coeff.shape)
        exp_info = exp_coeff @ self.exp_base.T
        face_shape = self.mean_shape.reshape([1, -1]) + id_info + exp_info

        return face_shape


class BFMModelPyTorch(object):
    def __init__(self, 
                 BFM_model_path="BFM/BFM_model_front.mat", 
                 recenter=True) -> None:
        model = loadmat(BFM_model_path)
        self.mean_shape = torch.from_numpy(model['meanshape'].astype(np.float32)) # [3*N,1]
        self.id_base = torch.from_numpy(model['idBase'].astype(np.float32)) # [3*N,80]
        self.exp_base = torch.from_numpy(model['exBase'].astype(np.float32)) # [3*N,64]
        self.key_point = torch.from_numpy(np.squeeze(model['keypoints']).astype(np.int64) - 1)
        
        if recenter:
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape = mean_shape - torch.mean(mean_shape, dim=0, keepdims=True)
            self.mean_shape = mean_shape.reshape([-1, 1])

    def compute_shape(self, id_coeff=None, exp_coeff=None):
        """Compute the complete 3D face shape

        Args:
            id_coeff (np.ndarray): (1, 80)
            exp_coeff (np.ndarray): (1, 64)

        Returns:
            np.ndarray: (B, 3*N)
        """
        disp = 0
        if id_coeff is not None:
            id_info = id_coeff @ self.id_base.T
            disp += id_info
        
        if exp_coeff is not None:
            exp_info = exp_coeff @ self.exp_base.T
            disp += exp_info
        
        face_shape = self.mean_shape.reshape([1, -1]) + disp
        return face_shape
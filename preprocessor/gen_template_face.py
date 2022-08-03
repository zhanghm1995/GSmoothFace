'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-23 19:18:36
Email: haimingzhang@link.cuhk.edu.cn
Description: Generate a template face for each face video
'''

import os
import os.path as osp
import numpy as np
from scipy.io import loadmat
from glob import glob


class BFMModel(object):
    def __init__(self, BFM_model_path, recenter=True) -> None:
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
            np.ndarray: (N, 3)
        """
        if id_coeff is None:
            id_coeff = np.zeros((1, 80)).astype(self.id_base.dtype)
        if exp_coeff is None:
            exp_coeff = np.zeros((1, 64)).astype(self.exp_base.dtype)

        id_info = id_coeff @ self.id_base.T
        exp_info = exp_coeff @ self.exp_base.T
        face_shape = self.mean_shape.reshape([1, -1]) + id_info + exp_info
        face_shape = face_shape.reshape([-1, 3])
        return face_shape


def write_obj(file, points, rgb=False):
    """Write obj file which can be opened by MeshLab

    Args:
        points (np.ndarray): (N, 3)
        file (str|path): save path
        rgb (bool, optional): including rgb information. Defaults to False.
    """
    fout = open(file, 'w')
    for i in range(points.shape[0]):
        if not rgb:
            fout.write('v %f %f %f %d %d %d\n' % (
                points[i, 0], points[i, 1], points[i, 2], 255, 255, 0))
        else:
            fout.write('v %f %f %f %d %d %d\n' % (
                points[i, 0], points[i, 1], points[i, 2], points[i, -3] * 255, points[i, -2] * 255,
                points[i, -1] * 255))


def main_gen_mean(data_root, vid_dir_list, save_template_obj=False):
    bfm_model = BFMModel("../data/BFM/BFM_model_front.mat")

    for vid_dir in vid_dir_list:
        video_path = osp.join(data_root, vid_dir)
        
        id_params_list = []
        ## 1) Get all 3d face parameters
        all_face_3d_params_paths = sorted(glob(osp.join(video_path, "deep3dface", "*.mat")))

        for face_3d_params_path in all_face_3d_params_paths:
            face_params_dict = loadmat(face_3d_params_path)
            id_coeff = face_params_dict['id']
            id_params_list.append(id_coeff)
        
        id_params_arr = np.concatenate(id_params_list, axis=0)

        ## 2) Compute the mean values
        mean_id_param = np.mean(id_params_arr, axis=0, keepdims=True)

        template_face = bfm_model.compute_shape(id_coeff=mean_id_param)
        
        if save_template_obj:
            write_obj(f"./template/{vid_dir}_template.obj", template_face)

        ## Save the template information
        template_face_path = osp.join(video_path, "template_face.npy")
        np.save(template_face_path, template_face) # (35709, 3)
        
        ## Save the id_coeff information
        id_coeff_path = osp.join(video_path, "id_coeff.npy")
        np.save(id_coeff_path, id_coeff) # (1, 80)


if __name__ == "__main__":
    data_root = "../data/HDTF_face3dmmformer"
    split = "small_train"

    all_videos_dir = open(osp.join(data_root, f'{split}.txt')).read().splitlines()

    main_gen_mean(data_root, all_videos_dir)

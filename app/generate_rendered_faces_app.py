'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-07-21 14:16:21
Email: haimingzhang@link.cuhk.edu.cn
Description: Generate the rendered faces for the 3DMM
'''

import sys

from cv2 import split
sys.path.append('.')
sys.path.append('../')
import os
import os.path as osp
from scipy.io import loadmat
import torch
import numpy as np
import cv2
from tqdm import tqdm
from visualizer.face_3d_visualizer import Face3DMMRenderer, loadmat2, get_coeff_vector


face_3dmm_renderer = Face3DMMRenderer()

def get_folder_list(root_dir):
    folder_list = [entry.path for entry in os.scandir(root_dir) if entry.is_dir()]
    folder_list = sorted(folder_list)
    return folder_list


def split_list(list_in, split_length=100):
    list_splited = [list_in[i:i + split_length] for i in range(0, len(list_in), split_length)]
    return list_splited

def process_single_folder(data_root, folder_name):
    deep3dface_dir = osp.join(data_root, folder_name, "deep3dface")
    
    face_3dmm_mat_list = sorted([fp.path for fp in os.scandir(deep3dface_dir) if fp.is_file() and fp.name.endswith(".mat")])
    print(len(face_3dmm_mat_list), face_3dmm_mat_list[:3])

    deep3dface_params_list, transform_params_list = [], []
    for deep3dface_mat_fp in tqdm(face_3dmm_mat_list):
        face_params_dict = loadmat(deep3dface_mat_fp)
        transform_params = loadmat2(deep3dface_mat_fp)['transform_params']

        coeff_matrix = torch.FloatTensor(get_coeff_vector(face_params_dict)) # (1, 257)
        deep3dface_params_list.append(coeff_matrix)
        
        transform_params_list.append(transform_params)
    
    deep3dface_params = torch.cat(deep3dface_params_list, dim=0) # (N, 257)
    transform_vec_params = np.stack(transform_params_list, axis=0)

    print(deep3dface_params.shape, transform_vec_params.shape)

    splited_deep3dface_params = split_list(deep3dface_params, split_length=100)

    rendered_image = face_3dmm_renderer.render_3dmm_face(
        deep3dface_params[None], transform_vec_params, need_save=False)
    print(rendered_image.shape)

    output_dir = "./temp"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(rendered_image.shape[0]):
        img = rendered_image[i]
        cv2.imwrite(osp.join(output_dir, f"{i:06d}.png"), img[..., ::-1])


def process_single_folder_v2(data_root, folder_name, save_data=True):
    deep3dface_dir = osp.join(data_root, folder_name, "deep3dface")
    
    face_3dmm_mat_list = sorted([fp.path for fp in os.scandir(deep3dface_dir) if fp.is_file() and fp.name.endswith(".mat")])

    deep3dface_params_list, transform_params_list = [], []
    for deep3dface_mat_fp in tqdm(face_3dmm_mat_list):
        face_params_dict = loadmat(deep3dface_mat_fp)
        transform_params = loadmat2(deep3dface_mat_fp)['transform_params']

        coeff_matrix = torch.FloatTensor(get_coeff_vector(face_params_dict)) # (1, 257)
        deep3dface_params_list.append(coeff_matrix)
        
        transform_params_list.append(transform_params)
    
    splited_deep3dface_params = split_list(deep3dface_params_list, split_length=100)
    splited_transform_params = split_list(transform_params_list, split_length=100)

    count = -1
    for batch_deep3dface, batch_trans_params in zip(splited_deep3dface_params, splited_transform_params):
        deep3dface_params = torch.cat(batch_deep3dface, dim=0) # (N, 257)
        transform_vec_params = np.stack(batch_trans_params, axis=0) # (N, 5)

        rendered_image = face_3dmm_renderer.render_3dmm_face(
            deep3dface_params[None], transform_vec_params, need_save=False)

        if save_data:
            output_dir = osp.join(data_root, folder_name, "deep3dface_512")
            os.makedirs(output_dir, exist_ok=True)

            for i in range(rendered_image.shape[0]):
                count += 1
                img = rendered_image[i]
                cv2.imwrite(osp.join(output_dir, f"{count:06d}.png"), img[..., ::-1])


def process_multiple_folders():
    data_root = "./HDTF_preprocessed"
    folder_list = get_folder_list(data_root)
    
    sub_folder_list = folder_list[180:200]
    print(sub_folder_list[:3])

    for folder in sub_folder_list:
        folder_name = osp.basename(folder)
        process_single_folder_v2(data_root, folder_name, save_data=True)

# process_single_folder("data/HDTF_preprocessed", "RD_Radio9_000")
# process_single_folder_v2("data/HDTF_face3dmmformer/train", "WDA_BarackObama_000")

if __name__ == "__main__":
    process_multiple_folders()


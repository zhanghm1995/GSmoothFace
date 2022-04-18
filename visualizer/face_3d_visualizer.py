'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-04-15 14:31:05
Email: haimingzhang@link.cuhk.edu.cn
Description: The visualizer class to visualize the 3DMM results
'''

import os
import numpy as np
import os.path as osp
from glob import glob
from tqdm import tqdm
from scipy.io import loadmat
import scipy.io as spio
import torch
from easydict import EasyDict
import cv2
from .face_3d_params_utils import get_coeff_vector, rescale_mask_V2
from .render_utils import MyMeshRender


def loadmat2(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        


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


class Face3DMMVisualizer(object):
    def __init__(self, deep3dface_dir, need_pose=True) -> None:
        """initialization function

        Args:
            deep3dface_dir (str|Path): the .mat file directory
            need_pose (bool, optional): whether rendering results with pose information. Defaults to True.
        """
        assert osp.exists(deep3dface_dir), f'{deep3dface_dir} does not exist.'
        
        self.face_3dmm_mat_list = sorted(glob(osp.join(deep3dface_dir, "*.mat")))
        self.need_pose = need_pose

        if need_pose:
            opt = EasyDict(center=112.0, focal=1015.0, z_near=5.0, z_far=15.0)
        else:
            opt = EasyDict(center=256.0, focal=1015.0, z_near=5.0, z_far=15.0)

        self.renderer = MyMeshRender(opt)

    def vis_3dmm_face(self, input, output_root=None):
        print("[INFO] Start visualization...")

        if isinstance(input, np.ndarray):
            input_array = input
        elif isinstance(input, torch.Tensor):
            input_array = input
        elif isinstance(input, str):
            assert osp.exists(input), f'{input} does not exist.'
            input_array = np.load(input) # (B, N)
        else:
            raise ValueError(f'{input} is not supported.')
        
        input_array = torch.FloatTensor(input_array)

        ## Get the minium length
        minimum_length = min(len(self.face_3dmm_mat_list), len(input_array))
        matrix_file_list = self.face_3dmm_mat_list[:minimum_length]
        input_array = input_array[:minimum_length, :]
        
        count = -1
        prog_bar = tqdm(matrix_file_list)
        for matrix_file in prog_bar:
            count += 1
            prog_bar.set_description(matrix_file)

            face_params_dict = loadmat(matrix_file)
            transform_params = loadmat2(matrix_file)['transform_params']

            curr_face_vertex = input_array[count:count+1, :]
            face_params_dict['exp'] = curr_face_vertex # use as expression

            if self.need_pose:
                coeff_matrix = torch.FloatTensor(get_coeff_vector(face_params_dict))
            else:
                coeff_matrix = torch.FloatTensor(get_coeff_vector(face_params_dict, reset_list=['trans', 'angle']))

            ret = self.renderer(coeff_matrix, None)
            image = self.renderer.compute_rendered_image()[0]

            ## Rescale the image to original size
            scaled_image = rescale_mask_V2(image, transform_params)

            if output_root is not None:
                file_name = osp.basename(matrix_file).replace(".mat", ".png")
                # cv2.imwrite(osp.join(output_root, file_name), masked_image)
                if self.need_pose:
                    scaled_image.save(osp.join(output_root, file_name))
                else:
                    cv2.imwrite(osp.join(output_root, file_name), image[..., ::-1])

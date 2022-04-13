'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-27 15:55:57
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import numpy as np


def get_coeff_vector(face_params_dict, reset_list=None):
    """Get coefficient vector from Deep3DFace_Pytorch results

    Args:
        face_params_dict (dict): the dictionary contains reconstructed 3D face

    Returns:
        [np.ndarray]: 1x257
    """
    keys_list = ['id', 'exp', 'tex', 'angle', 'gamma', 'trans']

    coeff_list = []
    for key in keys_list:
        if reset_list is not None and key in reset_list:
            value = np.zeros_like(face_params_dict[key])
            coeff_list.append(value)
        else:
            coeff_list.append(face_params_dict[key])
    
    coeff_res = np.concatenate(coeff_list, axis=1)
    return coeff_res

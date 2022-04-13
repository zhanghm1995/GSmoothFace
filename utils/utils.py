'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-22 21:49:37
Email: haimingzhang@link.cuhk.edu.cn
Description: Some useful functions
'''

import numpy as np
import torch


def get_loss_description_str(loss_dict):
    """Get the description string from a dictionary

    Args:
        loss_dict (dict): a dictionary with a name string and corresponding value

    Returns:
        str: a joined string
    """
    assert isinstance(loss_dict, dict)

    description_str = ""
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            description_str += f"{key}: {value.item():0.4f} "
        else:
            description_str += f"{key}: {value:0.4f} "
    return description_str


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array, range(0, 1), (3, H, W)
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.clamp(0.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def get_git_commit_id():
    """Get the Git commit hash id for logging usage

    Returns:
        str: hash id
    """
    import git
    repo = git.Repo(search_parent_directories=False)
    sha = repo.head.object.hexsha
    return sha
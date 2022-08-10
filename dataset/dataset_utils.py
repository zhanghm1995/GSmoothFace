'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-09 21:16:11
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import os
import os.path as osp
import torch
from PIL import Image


def load_split_file(root_dir, split):
    """Load the dataset split contents when given split name
    or the path to the split file.

    Args:
        root_dir (str|Path): _description_
        split (str|Path): split name or path to the split file.

    Returns:
        list: dataset split list.
    """
    if split.endswith('.txt'):
        assert osp.exists(split), f"{split} is not exist, please check!"
        split_fp = split
    else:
        split_fp = osp.join(root_dir, f'{split}.txt')

    content_list = open(split_fp).read().splitlines()
    return content_list


def read_image_sequence(data_root, video_dir, start_idx, fetch_length,
                        transform=None):
    source_image_list = []
    
    for idx in range(start_idx, start_idx + fetch_length):
        ## Read the face image and resize
        img_path = osp.join(data_root, video_dir, "face_image", f"{idx:06d}.jpg")
        source_image = Image.open(img_path).convert("RGB")
        
        if transform is not None:
            source_image = transform(source_image)
        
        source_image_list.append(source_image)

    img_seq_tensor = torch.stack(source_image_list) # to (T, 3, H, W)
    return img_seq_tensor
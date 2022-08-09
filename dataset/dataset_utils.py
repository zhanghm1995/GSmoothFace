'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-09 21:16:11
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import os
import os.path as osp


def load_split_file(root_dir, split):
    if osp.isfile(split):
        split_fp = split
    else:
        split_fp = osp.join(root_dir, f'{split}.txt')

    content_list = open(split_fp).read().splitlines()
    return content_list
'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-09-22 20:21:11
Email: haimingzhang@link.cuhk.edu.cn
Description: The the Face3DMMFormer model
'''

import sys
sys.path.append("./")
sys.path.append("../")

import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import time
import numpy as np
import torchvision
import cv2
from PIL import Image

import torch
from omegaconf import OmegaConf
from models.face_3dmm_former import Face3DMMFormer


config = OmegaConf.load("./config/face_3dmm_config.yaml")
model = Face3DMMFormer(config['Face3DMMFormer'])

bs = 16
face_3d_params = torch.randn((bs, 100, 64))
raw_audio = torch.randn((bs, 64000))

data_dict = {'raw_audio': raw_audio, 
             'gt_face_3d_params': face_3d_params}

model_output = model(data_dict, teacher_forcing=False)

for key, value in model_output.items():
    print(key, value.shape)


model_output = model.predict(data_dict)

for key, value in model_output.items():
    print(key, value.shape)
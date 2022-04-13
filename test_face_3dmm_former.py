'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-20 22:10:49
Email: haimingzhang@link.cuhk.edu.cn
Description: Test the Face3DMMFormer class
'''

import torch
from omegaconf import OmegaConf
from models.face_3dmm_former import Face3DMMFormer


config = OmegaConf.load("./config/face_3dmm_config.yaml")
model = Face3DMMFormer(config['Face3DMMFormer'])

face_3d_params = torch.randn((1, 100, 64))
raw_audio = torch.randn((1, 64000))

data_dict = {'raw_audio': raw_audio, 
             'gt_face_3d_params': face_3d_params}

model_output = model(data_dict, teacher_forcing=False)



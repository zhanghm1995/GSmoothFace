'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-09-23 10:53:08
Email: haimingzhang@link.cuhk.edu.cn
Description: The loss module for predicting face 3dmm parameters.
'''

import torch
import torch.nn as nn
import numpy as np


class Face3DMMLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config

        self.supervise_vertices = config.supervise_vertices

        if config.use_mouth_mask:
            binary_mouth_mask = np.load("./data/big_mouth_mask.npy")
            mouth_mask = np.ones(35709)
            mouth_mask[binary_mouth_mask] = 1.8
            self.mouth_mask_weight = torch.from_numpy(np.expand_dims(mouth_mask, 0)) # (1, 35709)

    def forward(self, data_dict, model_output):
        if self.supervise_vertices:
            vertice = data_dict['face_vertex'] # GT vertices

            pred_output = model_output['face_3d_params']

            exp_base = data_dict['exp_base'] # (1, 3N, 64)
            template = data_dict['template']
            vertice_out = template + torch.einsum('ijk,iak->iaj', exp_base, pred_output)

            if self.config.use_mouth_mask:
                batch, seq_len = vertice_out.shape[:2]
                ## If consider mouth region weight
                vertice_out = vertice_out.reshape((batch, seq_len, -1, 3))
                vertice = vertice.reshape((batch, seq_len, -1, 3))

                loss = torch.sum((vertice_out - vertice)**2, dim=-1) * self.mouth_mask_weight[None, ...].to(vertice)
                loss = torch.mean(loss)
        else:
            raise NotImplementedError
        return loss

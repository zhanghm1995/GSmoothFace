'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-06 14:57:55
Email: haimingzhang@link.cuhk.edu.cn
Description: Face to face model
'''

import torch
import torch.nn as nn

from .face_model import MappingNet, EditingNet


class Face2FaceGenerator(nn.Module):
    def __init__(
        self,
        mapping_net,
        editing_net,
        common):
        super(Face2FaceGenerator, self).__init__()
        self.mapping_net = MappingNet(**mapping_net)
        self.editing_net = EditingNet(**editing_net, **common)

    def forward(
        self, 
        input_image,
        rendered_image,
        driving_source=None, 
        stage=None
        ):
        output = {}
        descriptor = self.mapping_net(driving_source)
        output['fake_image'] = self.editing_net(input_image, rendered_image, descriptor)
        return output
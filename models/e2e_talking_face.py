'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-09 21:54:46
Email: haimingzhang@link.cuhk.edu.cn
Description: The end-to-end talking face generation model.
'''

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.nn import functional as F 

import cv2
import numpy as np

import kornia as K
from kornia import morphology as morph
from einops import rearrange

from util import instantiate_from_config
from visualizer.render_utils import MyMeshRender


def rescale_image(image, M, dsize=(512, 512)):
    image_rescaled = K.geometry.warp_affine(image, M, dsize=dsize)
    return image_rescaled


class TalkingFaceEnd2EndModel(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        self.config = config

        ## Define the 3DMM parameters estimation network
        self.face3dmm_pred_net = instantiate_from_config(config.face3dmm_pred_network)

        ## Define the face generation network
        self.face_generator = instantiate_from_config(config.face_generator)

        # ## Define the renderer for visualization
        self.face_renderer = MyMeshRender()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        self.kernel = torch.from_numpy(kernel.astype(np.int64))

        self.key_list = ['gt_face_origin_3d_params', 'reference_image', ]
    
    def forward(self, batch):
        """For inference

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        ## 1) Forward the 3DMM prediction network when given audio input
        model_output = self.face3dmm_pred_net.predict(batch) # (B, T, 64)
        model_output = F.pad(model_output, (0, 0, 0, 1), mode="replicate")
        
        model_output = rearrange(model_output, 'b t c -> (b t) c')

        ## 2) Get the blended image
        for key, value in batch.items():
            if torch.is_tensor(value) and key in self.key_list:
                batch[key] = rearrange(value, 'b t ... -> (b t) ...')

        blended_image = self._build_blended_image(batch, model_output)
        reference_image = batch['reference_image']

        face_gen_out_dict = self.face_generator(reference_image, blended_image)

        return face_gen_out_dict

    def training_step(self, batch, batch_idx):
        ## 1) Forward the 3DMM prediction network when given audio input
        if self.config.supervise_exp:
            pred_exp = self.face3dmm_pred_net(
                batch, self.criterion, teacher_forcing=self.config.teacher_forcing,
                return_loss=False, return_exp=True)
            face3dmm_pred_loss = self.criterion(pred_exp, batch['gt_face_3d_params'])
            face3dmm_pred_loss = torch.mean(face3dmm_pred_loss)
        else:
            face3dmm_pred_loss = self.model(
                batch, self.criterion, teacher_forcing=self.config.teacher_forcing)

        ## 2) Forward the face generation network when given the 3DMM parameters
        blended_image = self._build_blended_image(batch, )
        batch['blended_image'] = blended_image
        face_gen_out_dict = self.face_generator(batch)
        
        ## 3) Compute the loss

    
    def _build_blended_image(self, data_dict, pred_exp_params):
        face3dmm_params = data_dict['gt_face_origin_3d_params']

        curr_face3dmm_params = face3dmm_params[:, :257]
        curr_face3dmm_params[:, 80:144] = pred_exp_params

        rendered_face, rendered_mask = self.face_renderer.compute_rendered_face(
            curr_face3dmm_params, None, return_numpy=False)
        
        ## Fill the holes of rendered mask
        morpho_mask = morph.closing(rendered_mask, self.kernel.to(rendered_mask.device))

        rescaled_rendered_face = K.geometry.warp_affine(rendered_face, data_dict['trans_mat_inv'], dsize=(512, 512))
        rescaled_mask = K.geometry.warp_affine(morpho_mask, data_dict['trans_mat_inv'], dsize=(512, 512))
        
        rescaled_rendered_face = (rescaled_rendered_face - 0.5) / 0.5

        gt_face_image_seq = data_dict['gt_face_image_seq']

        blended_img_tensor = gt_face_image_seq * (1 - rescaled_mask) + \
                             rescaled_rendered_face * rescaled_mask
        
        return blended_img_tensor

    def get_visualization(self, data):
        pass

    
'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-24 11:03:02
Email: haimingzhang@link.cuhk.edu.cn
Description: Face2D3DFusionFormer class
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from wav2vec import Wav2Vec2Model
from .face_3dmm_one_hot_former import Face3DMMOneHotFormer
from .image_unet import ImageUNet, Generator


# Alignment Bias
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)


class Face2D3DFusionFormer(Face3DMMOneHotFormer):
    def __init__(self, args):
        super(Face2D3DFusionFormer, self).__init__(args)
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.config = args

        ## Define the model
        self.face_2d_net = Generator(dropout_p=0.2)
        self.face_2d_layer_norm = nn.LayerNorm(self.config.feature_dim)
        self.face_3d_layer_norm = nn.LayerNorm(self.config.feature_dim)

    def forward(self, data_dict, teacher_forcing=False):
        audio = data_dict['raw_audio']
        template = data_dict['template']
        vertice = data_dict['face_vertex']
        one_hot = data_dict['one_hot']
        gt_face_image = data_dict['gt_face_image'] # (B, S, 3, H, W)
        gt_mouth_mask_image = data_dict['gt_mouth_mask_image'] # (B, S, 1, H, W)
        gt_face_image_no_mouth = gt_face_image * (1 - gt_mouth_mask_image) # mouth masked face image

        self.device = audio.device
        batch_size, seq_len, _ = vertice.shape

        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        template = template.unsqueeze(1) # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)#(1, feature_dim)
        frame_num = vertice.shape[1]
        
        ## Extract the audio features
        hidden_states = self.audio_encoder(
            audio, self.dataset, frame_num=frame_num).last_hidden_state
        hidden_states = self.audio_feature_map(hidden_states)

        # ## Move to CPU to save GPU memory
        # data_dict['raw_audio'] = data_dict['raw_audio'].cpu()
        # data_dict['gt_face_image'] = data_dict['gt_face_image'].cpu()
        # data_dict['face_vertex'] = data_dict['face_vertex'].cpu()


        if teacher_forcing:
            vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
            style_emb = vertice_emb  
            vertice_input = torch.cat((template,vertice[:,:-1]), 1) # shift one position
            vertice_input = vertice_input - template
            vertice_input = self.vertice_map(vertice_input)
            vertice_input = vertice_input + style_emb
            vertice_input = self.PPE(vertice_input)
            vertice_input = self.face_3d_layer_norm(vertice_input)

            ## Get the 2D sequence input
            first_face_no_mouth_img = gt_face_image_no_mouth[:, 0:1, ...]
            # shifted_gt_mouth_img = torch.cat((first_face_no_mouth_img, gt_face_image[:, :-1, ...]), dim=1)

            ## Use the GT first frame
            shifted_gt_mouth_img = torch.cat((gt_face_image[:, 0:1, ...], gt_face_image[:, :-1, ...]), dim=1)

            face_input_img = torch.cat((shifted_gt_mouth_img, gt_face_image_no_mouth), dim=2) # (B, S, 6, H, W)
            face_2d_emb = self.face_2d_net.encode(face_input_img)
            face_2d_emb += style_emb
            face_2d_input = self.PPE(face_2d_emb)
            face_2d_input = self.face_2d_layer_norm(face_2d_input)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])

            tgt_mask = tgt_mask.repeat((batch_size, 1, 1))
            ## Repeat sth
            fusion_tgt_mask = tgt_mask.repeat((1, 2, 2))
            fusion_memory_mask = memory_mask.repeat((2, 2))
            fusion_input = torch.concat((face_2d_input, vertice_input), dim=1)
            fusion_hidden_states = hidden_states.repeat((1, 2, 1))

            fusion_out = self.transformer_decoder(
                fusion_input, fusion_hidden_states, tgt_mask=fusion_tgt_mask, memory_mask=fusion_memory_mask)
            seq_len = int(fusion_out.shape[1] // 2)
            face_2d_out = fusion_out[:, :seq_len, ...] # ()
            vertice_out = fusion_out[:, seq_len:, ...] # ()

            vertice_out = self.vertice_map_r(vertice_out)
            face_2d_out = self.face_2d_net.decode(face_2d_out) # to (B, S, 3, H, W)
        else:
            for i in range(frame_num):
                if i==0:
                    vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                    style_emb = vertice_emb
                    vertice_input = self.PPE(style_emb)

                    ## Handle the 2D data
                    # 1) Get the first frame GT face
                    first_face_img = gt_face_image[:, i:i+1, :, :, :] # (B, 1, 3, H, W)
                    first_face_no_mouth_img = gt_face_image_no_mouth[:, i:i+1, :, :, :]
                    first_input_img = torch.concat((first_face_img, first_face_no_mouth_img), dim=2) # (B, 1, 6, H, W)

                    # # NOTE: one batch could cause BatchNorm error!
                    # first_input_img = first_input_img.repeat((1, 2, 1, 1, 1)) # (B, 1, 6, H, W)

                    # 2) Encode the face image
                    face_2d_emb = self.face_2d_net.encode(first_input_img) # (B, 1, 1024)
                    face_2d_emb += style_emb
                    face_2d_input = self.PPE(face_2d_emb)
                else:
                    vertice_input = self.PPE(vertice_emb)

                    ## Get current frame gt face with masked mouth
                    face_no_mouth_img = gt_face_image_no_mouth[:, i:i+1, :, :, :]
                    input_img = torch.concat((face_2d_out[:, -1, :].unsqueeze(1), face_no_mouth_img), dim=2)
                    first_input_img = torch.concat((first_input_img, input_img), dim=1) # (B, S, 6, H, W)

                    # 2) Encode the face image
                    face_2d_emb = self.face_2d_net.encode(first_input_img) # (B, S, 1024)
                    face_2d_emb += style_emb
                    face_2d_input = self.PPE(face_2d_emb)

                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
                memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
                tgt_mask = tgt_mask.repeat((batch_size, 1, 1))
                ## Repeat sth
                fusion_tgt_mask = tgt_mask.repeat((1, 2, 2))
                fusion_memory_mask = memory_mask.repeat((2, 2))
                fusion_input = torch.concat((face_2d_input, vertice_input), dim=1)
                fusion_hidden_states = hidden_states.repeat((1, 2, 1))
                
                ## Apply the fusion transformer
                fusion_out = self.transformer_decoder(
                    fusion_input, fusion_hidden_states, tgt_mask=fusion_tgt_mask, memory_mask=fusion_memory_mask)

                seq_len = int(fusion_out.shape[1] // 2)
                face_2d_out = fusion_out[:, :seq_len, ...] # ()
                vertice_out = fusion_out[:, seq_len:, ...] # ()

                vertice_out = self.vertice_map_r(vertice_out)

                face_2d_out = self.face_2d_net.decode(face_2d_out) # to (B, S, 3, H, W)

                ## Build next input
                new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out + template
        return {"pred_face_vertex": vertice_out,
                'pred_face_image': face_2d_out}
        print(vertice_out.shape, face_2d_out.shape)
        # vertice_out = vertice_out + template
        # if self.config.use_mouth_mask:
        #     data_dict, seq_len = vertice_out.shape[:2]
        #     ## If consider mouth region weight
        #     vertice_out = vertice_out.reshape((data_dict, seq_len, -1, 3))
        #     vertice = vertice.reshape((data_dict, seq_len, -1, 3))

        #     loss = torch.sum((vertice_out - vertice)**2, dim=-1) * self.mouth_mask_weight[None, ...].to(vertice)
        #     loss = torch.mean(loss)
        # else:
        #     loss = criterion(vertice_out, vertice) # (data_dict, seq_len, V*3)
        #     loss = torch.mean(loss)
        # return loss

    def predict(self, audio, template, one_hot):
        self.device = audio.device

        template = template.unsqueeze(1) # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)
        hidden_states = self.audio_encoder(audio, self.dataset, output_fps=25).last_hidden_state
        if self.dataset == "BIWI":
            frame_num = hidden_states.shape[1]//2
        elif self.dataset == "vocaset":
            frame_num = hidden_states.shape[1]
        hidden_states = self.audio_feature_map(hidden_states)

        for i in range(frame_num):
            if i==0:
                vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                style_emb = vertice_emb
                vertice_input = self.PPE(style_emb)
            else:
                vertice_input = self.PPE(vertice_emb)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
            new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
            new_output = new_output + style_emb
            vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out + template
        return vertice_out


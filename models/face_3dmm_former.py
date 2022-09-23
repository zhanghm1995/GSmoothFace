'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-20 21:11:50
Email: haimingzhang@link.cuhk.edu.cn
Description: The model to generate 3DMM parameters from audio without consider the
ont-hot speaker vectors.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .wav2vec import Wav2Vec2Model


# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


# Alignment Bias
def enc_dec_mask(dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1)


# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Face3DMMFormer(nn.Module):
    def __init__(self, args, **kwargs):
        super(Face3DMMFormer, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.config = args

        ## Build the audio encoder
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_encoder.feature_extractor._freeze_parameters()

        self.audio_feature_map = nn.Linear(768, args.feature_dim)
        
        # motion encoder
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        
        # periodic positional encoding 
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period=args.period)

        # temporal bias
        self.biased_mask = init_biased_mask(n_head=4, max_seq_len=600, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, 
                                                   dim_feedforward=2*args.feature_dim, batch_first=True)        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # motion decoder
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)

        # style embedding
        self.obj_vector = nn.Linear(8, args.feature_dim, bias=False)
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, data_dict, teacher_forcing=True):
        audio = data_dict['raw_audio']
        face_3d_params = data_dict['gt_face_3d_params'] # (B, S, C)

        first_frame_params = face_3d_params[:, 0, :] # (B, C)

        device = audio.device

        batch_size, frame_num, _ = face_3d_params.shape

        ## Extract the audio features
        hidden_states = self._extract_audio_features(audio, frame_num)

        if teacher_forcing:
            vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
            style_emb = vertice_emb  
            vertice_input = torch.cat((template, face_3d_params[:,:-1]), 1) # shift one position
            vertice_input = vertice_input - template
            vertice_input = self.vertice_map(vertice_input)
            vertice_input = vertice_input + style_emb
            vertice_input = self.PPE(vertice_input)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=device)
            memory_mask = enc_dec_mask("vocaset", vertice_input.shape[1], hidden_states.shape[1]).to(device=device)
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
        else:
            for i in range(frame_num - 1):
                if i == 0:
                    vertice_emb = self.vertice_map(first_frame_params).unsqueeze(1) # (1, 1, feature_dim)
                
                vertice_input = self.PPE(vertice_emb)
                
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=device)
                tgt_mask = tgt_mask.repeat(batch_size, 1, 1)

                memory_mask = enc_dec_mask("vocaset", vertice_input.shape[1], hidden_states.shape[1]).to(device=device)
                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                vertice_out = self.vertice_map_r(vertice_out)

                new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = torch.cat((first_frame_params.unsqueeze(1), vertice_out), 1)
        return {'face_3d_params': vertice_out}

    @torch.no_grad()
    def predict(self, data_dict):
        audio = data_dict['raw_audio']
        device = audio.device

        hidden_states = self._extract_audio_features(audio, None)

        batch_size = audio.shape[0]
        dim_exp_params = self.config.vertice_dim
        
        frame_num = hidden_states.shape[1]

        first_frame_params = torch.zeros((batch_size, dim_exp_params)).to(device)

        for i in range(frame_num - 1):
            if i == 0:
                vertice_emb = self.vertice_map(first_frame_params).unsqueeze(1) # (1,1,feature_dim)
            
            vertice_input = self.PPE(vertice_emb)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=device)
            tgt_mask = tgt_mask.repeat(batch_size, 1, 1)

            memory_mask = enc_dec_mask("vocaset", vertice_input.shape[1], hidden_states.shape[1]).to(device=device)
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
            
            new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
            vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = torch.cat((first_frame_params.unsqueeze(1), vertice_out), 1)
        return vertice_out

    def _extract_audio_features(self, audio_input, output_num_frames):
        hidden_states = self.audio_encoder(
            audio_input, "vocaset", frame_num=output_num_frames, output_fps=25).last_hidden_state
        hidden_states = self.audio_feature_map(hidden_states)
        return hidden_states

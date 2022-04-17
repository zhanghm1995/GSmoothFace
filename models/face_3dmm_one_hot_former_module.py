'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-22 11:04:16
Email: haimingzhang@link.cuhk.edu.cn
Description: PL module to train model
'''

import torch
import numpy as np
import os
import os.path as osp
import torch.nn as nn
from scipy.io import wavfile
from torch.nn import functional as F 
import pytorch_lightning as pl
import subprocess

from .face_3dmm_one_hot_former import Face3DMMOneHotFormer


class Face3DMMOneHotFormerModule(pl.LightningModule):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()

        self.config = config

        self.save_hyperparameters()

        self.batch_size = self.config.dataset.batch_size

        ## Define the model
        self.model = Face3DMMOneHotFormer(config['Face3DMMFormer'])

        self.criterion = nn.MSELoss()

        if self.config.test_mode:
            from visualizer import Face3DMMVisualizer
            visualizer_config = self.config['visualizer']
            self.visualizer = Face3DMMVisualizer(visualizer_config.deep3dface_dir)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                     lr=self.config.lr, 
                                     weight_decay=self.config.wd,
                                     betas=(0.9, 0.999), 
                                     eps=1e-06)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, batch, criterion=None, teacher_forcing=True, return_loss=True, return_exp=False):
        return self.model(batch, 
                          criterion=criterion, 
                          teacher_forcing=teacher_forcing, 
                          return_loss=return_loss, 
                          return_exp=return_exp)

    def _run_step(self, batch):
        if self.config.supervise_exp:
            pred_exp = self.model(
                batch, self.criterion, teacher_forcing=self.config.teacher_forcing,
                return_loss=False, return_exp=True)
            loss = self.criterion(pred_exp, batch['gt_face_3d_params'])
            loss = torch.mean(loss)
        else:
            loss = self.model(
                batch, self.criterion, teacher_forcing=self.config.teacher_forcing)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._run_step(batch)

        ## Logging
        self.log('train/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss

        if self.config.supervise_exp:
            pred_exp = self.model(
                batch, self.criterion, teacher_forcing=self.config.teacher_forcing,
                return_loss=False, return_exp=True)
            loss = self.criterion(pred_exp, batch['gt_face_3d_params'])
            loss = torch.mean(loss)
        else:
            loss = self.model(
                batch, self.criterion, teacher_forcing=self.config.teacher_forcing)

        batch_size = audio.shape[0]
        
        ## Calcuate the loss
        self.log('train/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self._run_step(batch)
        
        ## Logging
        self.log('val/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        audio = batch['raw_audio']
        video_name = batch['video_name'][0]
        
        model_output = self.model.predict(batch)
        model_output = model_output.squeeze().detach().cpu().numpy() # (seq_len, 64)
        
        ## Create the saving directory
        save_dir = osp.join(self.logger.log_dir, "vis")
        if "/" in video_name:
            folder = video_name.split("/")[0]
            os.makedirs(osp.join(save_dir, folder), exist_ok=True)
        else:
            os.makedirs(save_dir, exist_ok=True)
        
        ## Save model output
        file_name = f"{video_name}_{batch_idx:03d}"

        np.save(osp.join(save_dir, f"{file_name}.npy"), model_output)

        ## Save audio
        audio_data = audio[0].cpu().numpy()
        wavfile.write(osp.join(save_dir, f"{file_name}.wav"), 16000, audio_data)

        ## Save rendered face images
        vis_dir = osp.join(save_dir, file_name)
        os.makedirs(vis_dir, exist_ok=True)
        self.visualizer.vis_3dmm_face(model_output, output_root=vis_dir)

        ## Save the audio file
        audio_file_path = osp.join(save_dir, f"{file_name}.wav")
        command = f"cp {audio_file_path} {vis_dir}/audio.wav"
        subprocess.call(command, shell=True)

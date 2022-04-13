'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-20 21:58:34
Email: haimingzhang@link.cuhk.edu.cn
Description: The Pytorch-Lightning Face3DMMFormer module
'''

import torch
import numpy as np
import os
import os.path as osp
from torch.nn import functional as F 
import pytorch_lightning as pl
from .face_3dmm_former import Face3DMMFormer


class Face3DMMFormerModule(pl.LightningModule):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()

        self.config = config

        self.save_hyperparameters()

        ## Define the model
        self.model = Face3DMMFormer(config['Face3DMMFormer'])
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd,
                                     betas=(0.9, 0.999), eps=1e-06)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        model_output = self.model(batch, teacher_forcing=False)
        
        ## Calcuate the loss
        loss_dict = self.compute_loss(batch, model_output)
        total_loss = 0.0
        for value in loss_dict.values():
            total_loss += value
        
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss
        
    def validation_step(self, batch, batch_idx):
        model_output = self.model(batch, teacher_forcing=False)
        
        ## Calcuate the loss
        loss_dict = self.compute_loss(batch, model_output)
        total_loss = 0.0
        for value in loss_dict.values():
            total_loss += value
        
        self.log('val/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def compute_loss(self, data_dict, model_output):
        loss_dict = {}
        
        ## 3D loss
        pred_params = model_output['face_3d_params']
        tgt_params = data_dict['gt_face_3d_params']

        loss_3dmm = 20 * F.smooth_l1_loss(pred_params[:, :, :], tgt_params[:, :, :])

        loss_dict['loss_3dmm'] = loss_3dmm
        return loss_dict

    def test_step(self, batch, batch_idx):
        model_output = self.model.predict(batch)
        
        log_dir = osp.join(self.logger.log_dir, "pred")
        os.makedirs(log_dir, exist_ok=True)
        
        ## Save prediction to files
        pred = model_output['face_3d_params'].cpu().numpy() # (B, S, 64)
        for batch in range(pred.shape[0]):
            seq_pred = pred[batch]
            file_path = osp.join(log_dir, f"{batch_idx:03d}-{batch}.npy")
            np.save(file_path, seq_pred)
            
            
        return model_output

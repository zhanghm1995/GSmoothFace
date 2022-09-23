'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-20 21:58:34
Email: haimingzhang@link.cuhk.edu.cn
Description: The Pytorch-Lightning Face3DMMFormer module
'''

import torch
import torch.nn as nn
import numpy as np
import os
import os.path as osp
from torch.nn import functional as F 
import pytorch_lightning as pl
from .face_3dmm_former import Face3DMMFormer
from .losses.face3dmm_loss import Face3DMMLoss
from visualizer import Face3DMMRenderer


class Face3DMMFormerModule(pl.LightningModule):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()

        self.config = config
        self.batch_size = self.config.dataset.batch_size

        self.save_hyperparameters()

        ## Define the model
        self.model = Face3DMMFormer(config.model.params)

        self.compute_loss = Face3DMMLoss(config.face3dmm_loss)

        self.face_3dmm_renderer = Face3DMMRenderer()

    def training_step(self, batch, batch_idx):
        model_output = self.model(batch, teacher_forcing=False)
        
        ## Calcuate the loss
        loss = self.compute_loss(batch, model_output)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        model_output = self.model(batch, teacher_forcing=False)
        
        ## Calcuate the loss
        loss = self.compute_loss(batch, model_output)

        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if batch_idx % 100 == 0:
            self._save_visualization(batch, batch_idx)
        return loss

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

    def _save_visualization(self, batch, batch_idx, save_dir=None):
        model_output = self.model.predict(batch).detach() # (B, T, 64)
        # model_output = F.pad(model_output, (0, 0, 0, 1), mode="replicate")

        face3dmm_params = batch['gt_face_origin_3d_params'][..., :257]
        face3dmm_params[:, :, 80:144] = model_output

        if save_dir is None:
            save_dir = osp.join(self.logger.log_dir, "results/val", f"epoch_{self.current_epoch:03d}")
        
        self.face_3dmm_renderer.render_3dmm_face(face3dmm_params, 
                                                 output_dir=save_dir,
                                                 rgb_mode=True,
                                                 name=batch_idx,
                                                 audio_array=batch['raw_audio'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd,
                                     betas=(0.9, 0.999), eps=1e-06)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

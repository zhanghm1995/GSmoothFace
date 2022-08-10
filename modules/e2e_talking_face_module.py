'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-09 21:42:06
Email: haimingzhang@link.cuhk.edu.cn
Description: End-to-end talking face generation PL module.
'''

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import pytorch_lightning as pl

from util import instantiate_from_config
from models.e2e_talking_face import TalkingFaceEnd2EndModel
from models.losses.perceptual import PerceptualLoss


class TalkingFaceEnd2EndModule(pl.LightningModule):
    def __init__(self, hp, **kwargs) -> None:
        super().__init__()

        self.hp = hp

        self.model = TalkingFaceEnd2EndModel(hp, **kwargs)
    
    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        ## Forward the training step
        output_dict = self.model(batch, batch_idx)

        ## Compute the loss
        fake_img = output_dict['fake_image']
        gt_image = batch['gt_face_image_seq']

        self.gen_losses = {}
        self.gen_losses["gen_perceptual_loss"] = self.criteria['perceptual_final'](fake_img, gt_image)
        self.gen_losses['gen_l1_loss'] = self.criteria['l1_loss'](fake_img, gt_image)

        total_loss = 0
        for key in self.gen_losses:
            self.gen_losses[key] = self.gen_losses[key] * self.weights[key]
            total_loss += self.gen_losses[key]

        self.gen_losses['total_loss'] = total_loss
        

    def validation_step(self):
        pass
    
    def _init_loss(self, opt):
        self._assign_criteria("gen_perceptual_loss", 
                              PerceptualLoss(**opt.perceptual_loss.params), 
                              opt.perceptual_loss.weight)
        self._assign_criteria("gen_l1_loss", nn.L1Loss(), opt.l1_loss.weight)

        self.face3dmm_pred_loss = instantiate_from_config(opt.face3dmm_pred_loss)

    def _assign_criteria(self, name, criterion, weight):
        self.criteria[name] = criterion
        self.weights[name] = weight

    def configure_optimizers(self):
        optimizer_list = []

        optimizer_g = instantiate_from_config(self.hp.generator.optimizer, params={"params": self.generator.parameters()})
        if "scheduler" in self.hp.generator:
            scheduler_g = instantiate_from_config(self.hp.generator.scheduler, params={"optimizer": optimizer_g})
            optimizer_list.append({
                "optimizer": optimizer_g,
                "lr_scheduler": {
                    "scheduler": scheduler_g,
                    "interval": 'step',
                    "monitor": False,
                },
            })
        else:
            optimizer_list.append({"optimizer": optimizer_g})

        optimizer_d = instantiate_from_config(self.hp.discriminator.optimizer, params={"params": self.discriminator.parameters()})
        if "scheduler" in self.hp.discriminator:
            scheduler_d = instantiate_from_config(self.hp.discriminator.scheduler, params={"optimizer": optimizer_g})
            optimizer_list.append({
                "optimizer": optimizer_d,
                "lr_scheduler": {
                    "scheduler": scheduler_d,
                    "interval": 'step',
                    "monitor": False,
                },
            })
        else:
            optimizer_list.append({"optimizer": optimizer_d})

        return optimizer_list
'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-20 21:40:09
Email: haimingzhang@link.cuhk.edu.cn
Description: My main training script
'''

import argparse
import torch
import os.path as osp
import pytorch_lightning as pl
from dataset import get_3dmm_dataset
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from utils.utils import get_git_commit_id
from models import get_model


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./config/face_3dmm_config.yaml', help='the config file path')
    parser.add_argument('--gpu', type=int, nargs='+', default=(0, 1), help='specify gpu devices')
    parser.add_argument('--checkpoint_dir', type=str, nargs='?', const="work_dir2/debug")
    parser.add_argument('--checkpoint', type=str, default=None, help="the pretrained checkpoint path")
    parser.add_argument('--test_mode', action='store_true', help="whether is a test mode")

    args = parser.parse_args()
    config = OmegaConf.load(args.cfg)

    if args.checkpoint_dir is None: # use the yaml value if don't specify the checkpoint_dir argument
        args.checkpoint_dir = config.checkpoint_dir
    
    config.update(vars(args)) # override the configuration using the value in args

    try:
        config['commit_id'] = get_git_commit_id()
    except:
        print("[WARNING] Couldn't get the git commit id")
    
    print(OmegaConf.to_yaml(config, resolve=True))

    return config


config = parse_config()

## Create model
model = get_model(config['model_name'], config)

if config.checkpoint is None:
    print(f"[WARNING] Train from scratch!")
else:
    print(f"Load pretrained model from {config.checkpoint}")
    model = model.load_from_checkpoint(config.checkpoint, config=config)

if not config['test_mode']:
    print(f"{'='*25} Start Traning, Good Luck! {'='*25}")

    ## ======================= Training ======================= ##
    ## 1) Define the dataloader
    train_dataloader = get_3dmm_dataset(config['dataset'], split="small_train")
    print(f"The training dataloader length is {len(train_dataloader)}")

    val_dataloader = get_3dmm_dataset(config['dataset'], split='small_val')
    print(f"The validation dataloader length is {len(val_dataloader)}")

    trainer = pl.Trainer(gpus=1, default_root_dir=config['checkpoint_dir'],
                         max_epochs=config.max_epochs,
                         check_val_every_n_epoch=config.check_val_every_n_epoch)
    
    # trainer = pl.Trainer(gpus=2, default_root_dir=config['checkpoint_dir'], 
    #                      accelerator="gpu", 
    #                      strategy="ddp",
    #                      max_epochs=config.max_epochs,
    #                      check_val_every_n_epoch=config.check_val_every_n_epoch)

    ## Resume the training state
    predictions = trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=config.checkpoint)
else:
    print(f"{'='*25} Start Testing, Good Luck! {'='*25}")

    # config['dataset']['audio_path'] = "data/audio_samples/obama.wav"
    config['dataset']['audio_path'] = "data/id00002/obama_weekly_029/obama_weekly_029.wav"
    config['dataset']['video_path'] = "data/id00002/obama_weekly_029/face_image"
    config['dataset']['face_3d_params_path'] = "data/id00002/obama_weekly_029/deep3dface.npz"

    test_dataloader = get_test_2d_3d_dataset(config['dataset'])
    print(f"The testing dataloader length is {len(test_dataloader)}")

    trainer = pl.Trainer(gpus=1, default_root_dir=config['checkpoint_dir'], logger=None)
    trainer.test(model, test_dataloader)
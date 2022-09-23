'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-24 19:29:46
Email: haimingzhang@link.cuhk.edu.cn
Description: Test the 2D 3D Dataset
'''

import sys
sys.path.append("./")
sys.path.append("../")

import torch

from omegaconf import OmegaConf
from dataset import get_dataset, get_test_dataset
from tqdm import tqdm
from PIL import Image
import numpy as np
import time
import cv2

import kornia as K
from kornia import morphology as morph
from visualizer.render_utils import MyMeshRender
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def test_2d_3d_dataset():
    config = OmegaConf.load("./config/face_3dmm_motion_mouth_mask_pix2pixhd.yaml")

    dataset_config = config['dataset']

    train_dataloader = get_dataset(dataset_config, split="voca_train", shuffle=False)
    print(len(train_dataloader))

    dataset = next(iter(train_dataloader))
    print(dataset['video_name'])

    print(dataset["gt_face_image"].shape)
    masked_face_image = dataset['gt_masked_face_image'][0, 0] * 255.0
    masked_face_image = masked_face_image.permute(1, 2, 0).numpy()

    image_numpy = masked_face_image.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save("./example_masked_gt3.png")

    masked_face_image = dataset['gt_face_image'][0, 0] * 255.0
    masked_face_image = masked_face_image.permute(1, 2, 0).numpy()

    image_numpy = masked_face_image.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save("./example_gt3.png")

    # for i, dataset in tqdm(enumerate(train_dataloader)):
    #     print(dataset["video_name"])


def test_Face3DMMOneHotDataset():
    from dataset.face_3dmm_one_hot_dataset import Face3DMMOneHotDataset

    data_root = "./data/HDTF_preprocessed"
    split = "./data/train.txt"
    
    dataset = Face3DMMOneHotDataset(data_root, split, fetch_length=40, need_load_image=True)
    print(len(dataset))

    item = dataset[0]
    print(item['one_hot'].shape)
    print(item['face_vertex'].shape)
    print(item['template'].shape)

    gt_img = item['gt_face_image_seq']
    print(gt_img.min(), gt_img.max())

    for key, value in item.items():
        if torch.is_tensor(value):
            print(key, value.shape)
        else:
            print(key, value)
    return item
    # start = time.time()
    # for i in tqdm(range(100)):
    #     item = dataset[i]
    # end = time.time()
    # print(f"{end - start} seconds", f"average time is {(end - start) / 100}")


def build_blended_image(data_dict):
    face_renderer = MyMeshRender()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    kernel = torch.from_numpy(kernel.astype(np.int64))

    face3dmm_params = data_dict['gt_face_origin_3d_params']

    curr_face3dmm_params = face3dmm_params[:, :257]

    rendered_face, rendered_mask = face_renderer.compute_rendered_face(
        curr_face3dmm_params, None, return_numpy=False)
    
    rendered_face = rendered_face.to("cpu")
    rendered_mask = rendered_mask.to("cpu")

    ## Fill the holes of rendered mask
    morpho_mask = morph.closing(rendered_mask, kernel.to(rendered_mask.device))

    rescaled_rendered_face = K.geometry.warp_affine(rendered_face, data_dict['trans_mat_inv'], dsize=(512, 512))
    rescaled_mask = K.geometry.warp_affine(morpho_mask, data_dict['trans_mat_inv'], dsize=(512, 512))
    rescaled_rendered_face = (rescaled_rendered_face - 0.5) / 0.5

    gt_face_image_seq = data_dict['gt_face_image_seq']

    blended_img_tensor = gt_face_image_seq * (1 - rescaled_mask) + \
                         rescaled_rendered_face * rescaled_mask
    
    res_dict = {}
    res_dict['blended_image'] = blended_img_tensor # (T, 3, 512, 512)

    vis_image = torch.concat([gt_face_image_seq, blended_img_tensor], dim=-1)
    
    vis_image = (vis_image + 1.0) / 2.0
    torchvision.utils.save_image(vis_image, "hdtf_test.jpg", padding=0, nrows=4)
    return res_dict


def test_Face3DMMTestDataset():
    from dataset import get_test_dataset
    
    config = OmegaConf.load("./config/AAAI/speaker_341_no_template_face.yaml")

    test_dataset = get_test_dataset(config.dataset)
    print(len(test_dataset))

    data = iter(test_dataset).next()
    print(data['one_hot'].shape)


def test_get_dataset_new():
    from dataset import get_dataset_new

    config = OmegaConf.load("config/CVPR/face3dmmformer_wo_speaker_id.yaml")

    dataset = get_dataset_new(config.dataset.train)
    print(len(dataset))

    entry = dataset[10]
    for key, value in entry.items():
        try:
            print(key, value.shape)
        except:
            print(key, type(value))


if __name__ == "__main__":
    test_get_dataset_new()
    exit(0)

    data = test_Face3DMMOneHotDataset()

    build_blended_image(data)

'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-24 19:29:46
Email: haimingzhang@link.cuhk.edu.cn
Description: Test the 2D 3D Dataset
'''

from omegaconf import OmegaConf
from dataset import get_dataset
from tqdm import tqdm
from PIL import Image
import numpy as np


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
    split = "./train.txt"
    
    dataset = Face3DMMOneHotDataset(data_root, split)
    print(len(dataset))

    item = dataset[0]
    print(item['one_hot'].shape)
    print(item['face_vertex'].shape)
    print(item['template'].shape)


if __name__ == "__main__":
    test_Face3DMMOneHotDataset()
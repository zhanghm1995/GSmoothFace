'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-20 20:53:40
Email: haimingzhang@link.cuhk.edu.cn
Description: Test Face3DMMDataset
'''

from omegaconf import OmegaConf
from easydict import EasyDict
from dataset import get_3dmm_dataset, get_test_dataset


def test_face_3dmm_dataset():
    config = EasyDict(data_root="./data/HDTF_preprocessed", fetch_length=75, batch_size=2, number_workers=4)

    train_dataloader = get_3dmm_dataset(config, split="train")
    print(len(train_dataloader))

    dataset = next(iter(train_dataloader))

    for key, value in dataset.items():
        print(key)
        print(value.shape)

def test_face_3dmm_dataset_loop():
    config = OmegaConf.load("./config/face_3dmm_config.yaml")

    train_dataloader = get_3dmm_dataset(config['dataset'], split="voca_train", shuffle=True)
    print(len(train_dataloader))

    dataset = next(iter(train_dataloader))

    print(dataset["video_name"])
    # for i, dataset in enumerate(train_dataloader):
    #     print(i)
    #     if i == 35:
    #         print("=========")
    #     for key, value in dataset.items():
    #         print(key)
    #         print(value.shape)


def test_face_3dmm_test_dataset():
    config = OmegaConf.load("./config/face_test_dataset.yaml")
    dataset_config = config['dataset']

    test_dataloader = get_test_dataset(dataset_config)
    print(len(test_dataloader))

    dataset = next(iter(test_dataloader))

if __name__ == "__main__":
    test_face_3dmm_test_dataset()
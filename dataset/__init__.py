import torch
from torch.utils.data import DataLoader

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def get_3dmm_dataset(config, shuffle=None):
    """Get the dataset contains 2D image and 3D information

    Args:
        config (dict): config parameters
        split (str): train or val
        shuffle (bool, optional): Whether shuffle. Defaults to None.

    Returns:
        DataLoader: the torch dataloader
    """
    if config.dataset_name == "Face3DMMDataset":
        from .face_3dmm_dataset import Face3DMMDataset
        dataset = Face3DMMDataset(data_root=config['data_root'], 
                                  split=split, 
                                  fetch_length=config['fetch_length'])
    elif config.dataset_name == "Face3DMMOneHotDataset":
        from .face_3dmm_one_hot_dataset import Face3DMMOneHotDataset
        dataset = Face3DMMOneHotDataset(**config)
    else:
        dataset_name = config.dataset_name
        raise ValueError(f"{dataset_name} dataset has not been defined")
    
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True if shuffle is None else shuffle,
        num_workers=config['number_workers'],
        # pin_memory=True,
        pin_memory=False,
        collate_fn=collate_fn
    )
    return data_loader


def get_dataset(config, split, shuffle=None):
    """Get the dataset contains 2D image and 3D information

    Args:
        config (dict): config parameters
        split (str): train or val
        shuffle (bool, optional): Whether shuffle. Defaults to None.

    Returns:
        DataLoader: the torch dataloader
    """
    if config.dataset_name == "Face2D3DDataset":
        from .face_2d_3d_dataset import Face2D3DDataset
        dataset = Face2D3DDataset(split=split, **config)
    elif config.dataset_name == "FaceDeep3DDataset":
        from .face_deep3d_dataset import FaceDeep3DDataset
        dataset = FaceDeep3DDataset(split=split, **config)
    else:
        dataset_name = config.dataset_name
        raise ValueError(f"{dataset_name} dataset has not been defined")

    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=("train" in split) if shuffle is None else shuffle,
        num_workers=config['number_workers'],
        # pin_memory=True,
        pin_memory=False,
        collate_fn=collate_fn
    )
    return data_loader

def get_test_dataset(config):
    from .face_3dmm_one_hot_test_dataset import Face3DMMTestDataset
    dataset = Face3DMMTestDataset(**config)

    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['number_workers'],
        collate_fn=collate_fn
    )
    return data_loader


def get_dataset_new(data_config):
    """Get the dataset contains 2D image and 3D information

    Args:
        config (dict): config parameters
        split (str): train or val
        shuffle (bool, optional): Whether shuffle. Defaults to None.

    Returns:
        DataLoader: the torch dataloader
    """
    if data_config.type == "Face2D3DDataset":
        from .face_2d_3d_dataset import Face2D3DDataset
        dataset = Face2D3DDataset(**data_config.params)
    elif data_config.type == "Face3DMMDataset":
        from .face_3dmm_dataset import Face3DMMDataset
        dataset = Face3DMMDataset(**data_config.params)
    else:
        dataset_name = data_config.type
        raise ValueError(f"{dataset_name} dataset has not been defined")

    return dataset


def get_dataloader_new(data_config):
    dataset = get_dataset_new(data_config)
    data_loader = DataLoader(
        dataset,
        **data_config.dataloader_params
    )
    return data_loader



def get_model(name, config):
    if name == "Face3DMMFormer":
        from .face_3dmm_former_module import Face3DMMFormerModule
        model = Face3DMMFormerModule(config)
    elif name == "Face3DMMOneHotFormer":
        from .face_3dmm_one_hot_former_module import Face3DMMOneHotFormerModule
        model = Face3DMMOneHotFormerModule(config)
    elif name == "Face2D3DFusionFormer":
        from .face_2d_3d_fusion_former_module import Face2D3DFusionFormerModule
        model = Face2D3DFusionFormerModule(config)
    elif name == "Face3DMMOneHotFormerPix2PixHDModule":
        from .face_3dmm_one_hot_former_pix2pixhd_module import Face3DMMOneHotFormerPix2PixHDModule
        model = Face3DMMOneHotFormerPix2PixHDModule(config)
    else:
        raise ValueError(f"{name} model has been defined!")
    
    return model
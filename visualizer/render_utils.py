'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-24 09:15:41
Email: haimingzhang@link.cuhk.edu.cn
Description: The utilities to render the 3DMM face
'''

import numpy as np
from numpy.lib.npyio import save
import torch
from torch import Tensor
import trimesh

from .nvdiffrast_utils import MeshRenderer
from .bfm import ParametricFaceModel


class MyMeshRender(object):
    def __init__(self, opt) -> None:
        super().__init__()

        self.device = torch.device('cuda') 

        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi

        self.renderer = MeshRenderer(
                rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(2 * opt.center)
            )

        self.facemodel = ParametricFaceModel(
                bfm_folder='../data/BFM', camera_distance=10.0, focal=opt.focal, center=opt.center,
                is_train=False, default_name='BFM_model_front.mat')

        self.facemodel.to(self.device)
        

    def __call__(self, face_params: Tensor, face_vertex: Tensor = None):
        """The forward function

        Args:
            face_params (Tensor): (B, 257)
            face_vertex (Tensor): (1, N)

        Returns:
            _type_: _description_
        """
        face_params = face_params.to(self.device)

        if face_vertex is not None:
            face_vertex = face_vertex.to(self.device)
        
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(face_params, face_vertex)

        ### Apply the face masking
        # print(self.pred_vertex.shape, self.pred_tex.shape, self.pred_color.shape)
        
        # print(self.pred_vertex.shape, torch.min(self.pred_vertex[0, :, 1]), torch.max(self.pred_vertex[0, :, 1]))

        # below_face_mask = self.pred_vertex.cpu().numpy()[0, :, 1] < -0.15

        # below_face_mask = ~np.load("big_mouth_mask.npy")

        # mouth_mask = (self.facemodel.face_seg_mask == 2)
        # self.pred_color[:, below_face_mask, :] = 0.0
        
        self.pred_mask, _, self.pred_face = self.renderer(
            self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)
        return None


    def compute_mesh(self, save_path=None):
        recon_shape = self.pred_vertex  # get reconstructed shape
        recon_shape[..., -1] = 10 - recon_shape[..., -1] # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        recon_color = self.pred_color
        recon_color = recon_color.cpu().numpy()[0]
        tri = self.facemodel.face_buf.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri, vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8))
        
        if save_path is not None:
            mesh.export(save_path)
        
        return mesh

    def compute_rendered_image(self):
        output_vis = self.pred_face

        output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
        output_vis_numpy_raw = output_vis_numpy_raw.astype(np.uint8)
        
        return output_vis_numpy_raw

'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-27 16:02:11
Email: haimingzhang@link.cuhk.edu.cn
Description: The main script to rendering the FaceFormer 3DMM prediction
'''

from PIL import Image
import numpy as np
from glob import glob
import os
import os.path as osp
from tqdm import tqdm
from scipy.io import loadmat, savemat
from face_3d_params_utils import get_coeff_vector
import cv2
import subprocess
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help='the data root path')
    parser.add_argument('--video_name', type=str, help='the video data name')
    parser.add_argument('--output_root', type=str, help='the output root path')
    parser.add_argument('--gen_vertex_path', type=str, help='generated face vertex file path')
    parser.add_argument('--need_pose', action='store_true', help='whether need rendering face with pose')

    args = parser.parse_args()
    return args


def get_contour(im):
    contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    out = np.zeros_like(im)
    out = cv2.fillPoly(out, contours, 255)

    return out


def get_masked_region(mask_img):
    gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    contour = get_contour(thresh)
    return contour


def rescale_mask(scaled_mask: np.array, transform_params: list) -> np.array:
    """
    Uncrops and rescales (i.e., resizes) the given scaled and cropped mask back to the
    resolution of the original image using the given transformation parameters.
    """
    
    # Parse transform params.
    original_image_width, original_image_height = transform_params[0:2]
    s = transform_params[2]  # the scaling parameter
    s = (s / 102.0) ** -1
    t = transform_params[3:].astype(np.float)  # some parameters for transformation
    t = [elem.item() for elem in t]
        
    # Repeat the computations for downscaling from preprocess_img.py/process_img() to get
    # the parameters needed for uncropping and rescaling the mask.
    
    # Get the width and height of the original image after downscaling.
    scaled_image_width = np.array((original_image_width / s*102)).astype(np.int32)  
    scaled_image_height = np.array((original_image_height / s*102)).astype(np.int32)
    
    scaled_mask_size = scaled_mask.shape[0]  # e.g. 224, NB. a scaled and cropped mask always has a square shape
    
    # Get an x or y coordinate for all sides (borders) of the mask.
    left_side_x = (scaled_image_width/2 - scaled_mask_size/2 + float((t[0] - original_image_width/2)*102/s)).astype(np.int32)
    right_side_x = left_side_x + scaled_mask_size
    upper_side_y = (scaled_image_height/2 - scaled_mask_size/2 + float((original_image_height/2 - t[1])*102/s)).astype(np.int32)
    lower_side_y = upper_side_y + scaled_mask_size
        
    # Compute the number of black ('missing') pixels to add to all sides of the mask.
    n_missing_pixels_left = left_side_x
    n_missing_pixels_right = scaled_image_width - right_side_x
    n_missing_pixels_top = upper_side_y
    n_missing_pixels_bottom = scaled_image_height - lower_side_y

    # Define np.arrays with the needed number of black pixels.
    if n_missing_pixels_left < 0:
        black_pixels_left = np.zeros(shape=(scaled_mask_size, 0, 3), dtype='uint8')
        scaled_mask = scaled_mask[:, -n_missing_pixels_left:]
    else:
        black_pixels_left = np.zeros(shape=(scaled_mask_size, n_missing_pixels_left, 3), dtype='uint8')

    if n_missing_pixels_right < 0:
        tmp = np.hstack([black_pixels_left, scaled_mask[:, :n_missing_pixels_right]])
        # tmp = np.hstack([black_pixels_left, scaled_mask])[:, :n_missing_pixels_right]
    else:
        black_pixels_right = np.zeros(shape=(scaled_mask_size, n_missing_pixels_right, 3), dtype='uint8')
        tmp = np.hstack([black_pixels_left, scaled_mask, black_pixels_right])

    if n_missing_pixels_top >= 0:
        black_pixels_top = np.zeros(shape=(n_missing_pixels_top, scaled_image_width, 3), dtype='uint8')
    else:
        black_pixels_top = np.zeros(shape=(0, scaled_image_width, 3), dtype='uint8')
        tmp = tmp[-n_missing_pixels_top:, :]

    if n_missing_pixels_bottom >= 0:
        black_pixels_bottom = np.zeros(shape=(n_missing_pixels_bottom, scaled_image_width, 3), dtype='uint8')
        uncropped_mask = np.vstack([black_pixels_top, tmp, black_pixels_bottom])
    else:
        # uncropped_mask = np.vstack([black_pixels_top, tmp])[:n_missing_pixels_bottom, :]
        uncropped_mask = np.vstack([black_pixels_top, tmp[:n_missing_pixels_bottom, :]])
    
    # Rescale (i.e., resize) the uncropped mask back to the resolution of the original image.
    uncropped_and_rescaled_mask = Image.fromarray(uncropped_mask).resize((original_image_width, original_image_height), 
                                                                          resample=Image.BICUBIC)
    
    return uncropped_and_rescaled_mask


def rescale_mask_V2(input_mask: np.array, transform_params: list):
    """
    Uncrops and rescales (i.e., resizes) the given scaled and cropped mask back to the
    resolution of the original image using the given transformation parameters.
    """
    original_image_width, original_image_height = transform_params[:2]
    s = np.float64(transform_params[2])
    t = transform_params[3:]
    target_size = 224.0

    scaled_image_w = (original_image_width * s).astype(np.int32)
    scaled_image_h = (original_image_height * s).astype(np.int32)
    left = (scaled_image_w/2 - target_size/2 + float((t[0] - original_image_width/2)*s)).astype(np.int32)
    up = (scaled_image_h/2 - target_size/2 + float((original_image_height/2 - t[1])*s)).astype(np.int32)

    # Parse transform params.
    mask_scaled = Image.new('RGB', (scaled_image_w, scaled_image_h), (0, 0, 0))
    mask_scaled.paste(Image.fromarray(input_mask), (left, up))
    
    # Rescale the uncropped mask back to the resolution of the original image.
    uncropped_and_rescaled_mask = mask_scaled.resize((original_image_width, original_image_height), 
                                                      resample=Image.CUBIC)
    return uncropped_and_rescaled_mask


import scipy.io as spio


def loadmat2(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def vis_rendered_face_list(args, data_root: str, output_root=None, need_pose=True):
    """Render the face by using the matrix generated by Deep3DFace_Pytorch repo

    Args:
        data_root (str): matrix file directory path
        output_root (str, optional): directory for saving rendered results. Defaults to None.
    """
    import torch
    import render_utils
    from easydict import EasyDict

    assert osp.exists(data_root), f'{data_root} does not exist.'

    if need_pose:
        opt = EasyDict(center=112.0, focal=1015.0, z_near=5.0, z_far=15.0)
    else:
        opt = EasyDict(center=256.0, focal=1015.0, z_near=5.0, z_far=15.0)

    renderer = render_utils.MyMeshRender(opt)

    matrix_file_list = sorted(glob(osp.join(data_root, "*.mat")))

    ## Read the generated face vertex
    gen_face_vertex = torch.FloatTensor(np.load(args.gen_vertex_path)) # (B, N)

    ## Get the minium length
    minimum_length = min(len(matrix_file_list), len(gen_face_vertex))
    matrix_file_list = matrix_file_list[:minimum_length]
    gen_face_vertex = gen_face_vertex[:minimum_length, :]
    print(f"minimum_length is {minimum_length}")

    count = -1
    prog_bar = tqdm(matrix_file_list)
    for matrix_file in prog_bar:
        count += 1
        prog_bar.set_description(matrix_file)

        face_params_dict = loadmat(matrix_file)
        transform_params = loadmat2(matrix_file)['transform_params']

        if count == 0:
            first_id_param = face_params_dict['id']
        # face_params_dict['id'] = first_id_param

        curr_face_vertex = gen_face_vertex[count:count+1, :]
        face_params_dict['exp'] = curr_face_vertex # use as expression

        if need_pose:
            coeff_matrix = torch.FloatTensor(get_coeff_vector(face_params_dict))
        else:
            coeff_matrix = torch.FloatTensor(get_coeff_vector(face_params_dict, reset_list=['trans', 'angle']))

        ret = renderer(coeff_matrix, None)
        image = renderer.compute_rendered_image()[0]

        ## Rescale the image to original size
        scaled_image = rescale_mask_V2(image, transform_params)

        ## Get the binary image
        # scaled_image = np.asarray(scaled_image)
        # masked_image = get_masked_region(scaled_image)

        if output_root is not None:
            file_name = osp.basename(matrix_file).replace(".mat", ".png")
            # cv2.imwrite(osp.join(output_root, file_name), masked_image)
            if need_pose:
                scaled_image.save(osp.join(output_root, file_name))
            else:
                cv2.imwrite(osp.join(output_root, file_name), image[..., ::-1])



if __name__ == "__main__":
    import shutil

    args = parse_args()
    
    data_root = osp.join(args.data_root, args.video_name, "deep3dface")
    output_root = args.output_root

    # if osp.exists(output_root):
    #     print("Delete the output folder...")
    #     shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    vis_rendered_face_list(args, data_root, output_root=output_root, need_pose=args.need_pose)


    ## Move the audio file
    audio_file_path = args.gen_vertex_path.replace(".npy", ".wav")
    command = f"cp {audio_file_path} {output_root}"
    print(command)
    subprocess.call(command, shell=True)
'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-24 21:07:19
Email: haimingzhang@link.cuhk.edu.cn
Description: The UNet-based image generation network
'''

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import segmentation_models_pytorch as smp # pip install segmentation-models-pytorch
from .conv import Conv2dTranspose, Conv2d


class ImageUNet(nn.Module):
    """Image tokenization encoder for input (192, 192) image
    """
    def __init__(self, image_size=192, in_ch=6):
        super(ImageUNet, self).__init__()

        self.image_size = image_size

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(in_ch, 16, kernel_size=15, stride=1, padding=7)), # 96*2,96*2      # kernel_size和 padding变大了

            nn.Sequential(Conv2d(16, 32, kernel_size=11, stride=2, padding=5), # 48*2,48*2      # kernel_size和 padding变大了
                        Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                        Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=7, stride=2, padding=3),    # 24*2,24*2      # kernel_size和 padding变大了
                        Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                        Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                        Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 12*2,12*2
                        Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                        Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # 6*2,6*2
                        Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                        Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),     # 3*2,3*2
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),


            ####################下面为新增加的1个下采样快######################
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 3,3                # 新增加的卷积层
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),    #新增加的卷积层
            # ######################上面为新增加的1个下采样快######################


            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
                          Conv2d(512, 1024, kernel_size=1, stride=1, padding=0)),])    #1*1

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),),              #  audio_embedding

            nn.Sequential(Conv2dTranspose(1536, 512, kernel_size=3, stride=1, padding=0), # 3,3    #(N-1)*S-2P+K    audio_embedding +
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

            ####################下面为新增加的1个 上采样快######################
            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),       #新增加的卷积层
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),                    #新增加的卷积层
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),  # 12, 12        #新增加的卷积层
            ####################上面为新增加的1个 上采样快######################


            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 96, 96

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),),]) # 192,192    
        
        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid()) 
        
        self.audio_fc = nn.Linear(128, 512)

    def encode(self, input: Tensor):
        """Encode the batched input image sequences into tokens

        Args:
            input (Tensor): (B, T, C, H, W)

        Returns:
            (Tensor): (B, T, C)
        """
        # input (B, T, C, H, W)
        B, T, C, H, W = input.shape
        input = input.reshape((-1, C, H, W))

        self.feats = []
        x = input
        for f in self.face_encoder_blocks:
            x = f(x)
            self.feats.append(x)

        ## Convert to (B, T, C)
        output = self.feats[-1] # 4-d tensor
        output = output.reshape(B, T, -1)
        return output

    def decode(self, input: Tensor):
        """Decode the embedding to whole image

        Args:
            input (Tensor): (B, T, C)

        Raises:
            e: _description_

        Returns:
            Tensor: (B, T, 3, H, W)
        """
        B, T, C = input.shape

        input = input.reshape(-1, C, 1, 1)
        x = input

        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, self.feats[-1]), dim=1)
            except Exception as e:
                raise e
            
            self.feats.pop()
        
        x = self.output_block(x) # (B, C, H, W)

        output = x.reshape((B, T, 3, self.image_size, self.image_size))
        return output

    def forward(self, audio_embedding, face_sequences):
        """Generate the image by using audio embedding and masked face sequences

        Args:
            audio_embedding (_type_): (B, S, E)
            face_sequences (Tensor): (B, T, 3, 224, 224)

        Raises:
            e: _description_
        """
        B, T, C, H, W = face_sequences.shape
        face_sequences = face_sequences.reshape((-1, C, H, W))

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)
        
        x = self.audio_fc(audio_embedding)[..., None, None]
        B, T, C = x.shape[:3]

        x = x.reshape(-1, C, 1, 1)
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                raise e
            
            feats.pop()
        
        x = self.output_block(x)
        output = x.reshape((B, T, 3, self.image_size, self.image_size))
        return output


class Generator(nn.Module):
    def __init__(self, dropout_p=0.4):
        super(Generator, self).__init__()
        self.dropout_p = dropout_p
        # Load Unet with Resnet34 Embedding from SMP library. Pre-trained on Imagenet
        self.unet = smp.Unet(encoder_name="resnet34", 
                             encoder_weights="imagenet", 
                             in_channels=6, classes=3, activation=None)
        # Adding two layers of Dropout as the original Unet doesn't have any
        # This will be used to feed noise into the networking during both training and evaluation
        # These extra layers will be added on decoder part where 2D transposed convolution is occured
        for idx in range(1, 3):
            self.unet.decoder.blocks[idx].conv1.add_module('3', nn.Dropout2d(p=self.dropout_p))
        # Disabling in-place ReLU as to avoid in-place operations as it will
        # cause issues for double backpropagation on the same graph
        for module in self.unet.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        
        self.enc_conv = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                      nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0))
        self.dec_conv = nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=0, output_padding=0),
                                      nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1))

    # def forward(self, x):
    #     x = self.unet(x)
    #     x = F.relu(x)
    #     return x

    def encode(self, x):
        # input (B, T, C, H, W)
        B, T, C, H, W = x.shape
        input = x.reshape((-1, C, H, W))

        self.feature_list = self.unet.encoder(input)
        last_feature = self.feature_list[-1]
        output = self.enc_conv(last_feature)
        return output.reshape(B, T, -1)
    
    def decode(self, x):
        B, T, C = x.shape

        input = x.reshape(-1, C, 1, 1)
        x = self.dec_conv(input)
        self.feature_list[-1] = x

        decoder_output = self.unet.decoder(*self.feature_list)
        masks = self.unet.segmentation_head(decoder_output)

        output = masks.reshape((B, T, 3, 192, 192))
        return output
# Data Preprocessor
In this document we will describe how to preprocess your dataset for training.

## Prepare the video
The video FPS should be 25fps. And make sure every frame has only one face in it.

## Crop the faces
You need to crop the faces from each video, please make sure the croped images with size 512x512, and the background is static. 
The face images are named in the `06d.jpg` format.

I obtained our training data from HDTF dataset.

## Extract the landmarks
In order to reconstruct the 3D face model, we need to extract the landmarks from the face images. You can run the following command to obtain the five landmarks and save them into txt files.
```bash
python prepare_data.py
```
Please modify the image path in the code to your own data path.

## Reconstruct the 3D face model
We use the [Deep3DFaceRecon_pytorch](https://github.com/zhanghm1995/Deep3DFaceRecon_pytorch) project to reconstruct the 3D face model. Due to we modify something from the original one, you need to use our modified version.

```bash
git clone https://github.com/zhanghm1995/Deep3DFaceRecon_pytorch.git
git checkout zhm_main
```
And then please read the `my_README.md` in the Deep3DFaceRecon_pytorch project to know how to use it.

After finishing the reconstruction, you will get the `*.mat` files which contains the 3D face model.

## Generate the template face
Since the reconstructed faces from Deep3DFaceRecon_pytorch directly would have noise, especially for video frames of the same identity. Here, we use a simple but effective way to alleviate the noise.

You can run the following command to get the template face and average identity coefficients for each video.
```bash
python gen_template_face.py
```
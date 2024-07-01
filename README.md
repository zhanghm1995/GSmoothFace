## GSmothFace: Generalized Smooth Talking Face Generation via Fine Grained 3D Face Guidance

## Installation
Python>=3.6
- Check the required python packages in `requirements.txt`.
- ffmpeg

```bash
conda create -n gsmoothface python=3.7
conda activate gsmoothface

pip install -r requirements.txt
conda install ffmpeg  # we need to install ffmpeg from anaconda to include the x264 encoder
```

For visualization, you need to install the [nvdiffrast](https://nvlabs.github.io/nvdiffrast/) package, please check the link to compile and install it in your python environment.
```bash
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
```

## Demo

### Data preparation
First of all, please prepare the driven audio you want to test and place them in `./data/audio_samples` folder. Here we provide some audio samples for you quickly testing in [BaiduNetDisk](https://pan.baidu.com/s/1v9nlpLgrrEZXy53bwKsvNQ) (code: uhg9).

### Run the demo script
Run the following command in the terminal. And before run the demo, please change the `--checkpoint_dir` in the script to specify the path of saved results. And please update the `checkpoint` in the script to specify the path of pretrained model.

```bash
bash demo.sh
```

The results will be saved to `<checkpoint_dir>/lightning_logs/version_*/vis/train/*` by default, containing the predicted expression parameters in `.npy` format, the rendered video without head pose in `.mp4` format. You can play this video for checking the lip-synchronization performance.


### Generate the video
After getting the rendered images, you can generate the video by the following command.

```bash
bash create_video_app.sh
```
And pay attention to the corresponding audio file is placed in the rendered images folder, you can specify this audio path when you run above command to generate the video with audio signal.

## Train
### Data preparation
1.Our method uses Basel Face Model 2009 (BFM09) to represent 3d faces (the 3DMM model). You can download them from [here](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads) (register and download). And download the Expression Basis (`Exp_Pca.bin`) from this [link (google drive)](https://drive.google.com/file/d/1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6/view). Download the BFM model Front file `BFM_model_front.mat` from this [link](https://drive.google.com/drive/folders/1YCxXKJFfo1w01PzayhnxWSZZK5k7spSH). After downloading all above files, link the BFM folder in the `data` directory, the file strutures are as follows:
```bash
./data/BFM
├── 01_MorphableModel.mat
├── BFM_exp_idx.mat
├── BFM_front_idx.mat
├── BFM_model_front.mat
├── Exp_Pca.bin
├── facemodel_info.mat
├── select_vertex_id.mat
├── similarity_Lm3D_all.mat
└── std_exp.txt
```
Actually, we only use the `BFM_model_front.mat` file for training GSmoothFace, but you also need other files for preparing the training dataset.

2.Prepare the talking face videos for training and testing. Our method supports the [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) dataset and [HDTF](https://github.com/MRzzm/HDTF) dataset. Due to the datasets license, we cannot provide the original data directly. You can download them and process it by yourself, and the preprocessing code is in [here](https://github.com/zhanghm1995/Deep3DFaceRecon_pytorch).

Here we provide a **small** portion of preprocessed HDTF dataset for quick start. You can download it from [BaiduNetDisk](https://pan.baidu.com/s/1v9nlpLgrrEZXy53bwKsvNQ) (code: uhg9).
After downloading, link the HDTF folder in the `data` directory, the file struture is as follows:
```bash
./data/HDTF_face3dmmformer
├── train
│   ├── WDA_BarackObama_000
│   ├── WDA_CarolynMaloney1_000
│   ├── WDA_ChrisMurphy0_000
│   ├── WDA_ChrisVanHollen1_000
│   ├── WDA_JeanneShaheen0_000
│   ├── WDA_KimSchrier_000
│   ├── WRA_DanSullivan_000
│   └── WRA_KellyAyotte_000
```

### Train models
GSmoothFace consists of two models: 1) The generic `A2EP` model, which is based on the 3DMM parameters inferred by Deep3DFace model and the Wav2Vec 2.0 model, learning to map the raw audio inputs to corresponding 3DMM parameters; 2) another generic GAN-based `TAFT` model trained on datasets with paired blended image and reference image. 

**Note**: If you have network connection problems when downloading the the Wav2Vec 2.0 pretrained models, you can manually download them from [huggingface](https://huggingface.co/facebook/wav2vec2-base-960h).

#### Step 1: Train the `A2EP` model
```bash
python main_train_one_hot.py --cfg config/gsmoothface_small.yaml
```

#### Step 2: Train the `TAFT` model
```bash
git clone https://github.com/zhanghm1995/GSmoothFace_TAFT

cd GSmoothFace_TAFT
bash train.sh
```

#### Step 3: Inference
You can infer the GSmoothFace with the following steps:

Step 3.1: Predict the 3DMM parameters conform to your given audio samples. Here we also provide some audio samples for you quickly testing in the above mentioned [BaiduNetDisk](https://pan.baidu.com/s/1v9nlpLgrrEZXy53bwKsvNQ) (code: uhg9). Please modify the `audio_path` and the `video_name` in the `config/demo.yaml` file with your driven audio and the speaker you want to generate. Then run the following command:
```bash
python demo.py --cfg config/demo.yaml --test_mode \
               --checkpoint <your/checkpoint/path> \
               --checkpoint_dir work_dir/demo
```
After running, you will obtain the corresponding 3DMM parameters in `.npy` format as well as the rendered 3DMM face videos without head pose in the folder `work_dir/demo/lighting_logs/version_0/vis` by default. And because our method slice the long audio with same length, the results will be divided into multiple parts.

Step 3.2: Render the predicted 3DMM parameters with head poses.
```bash
bash app/main_render_pred_results.sh
```
The rendered results will be placed in the `<output_root>` specified by yourself. You can also make a sound video with our provided script:
```bash
bash create_video_app.sh
```

Step 3.3: Generate the final talking face video via the [TAFT](https://github.com/zhanghm1995/GSmoothFace_TAFT) module.
```bash
cd GSmoothFace_TAFT
bash demo.sh
```

## Acknowledgement

We gratefully acknowledge the [Deep3DFace_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch), [FaceFormer](https://github.com/EvelynFan/FaceFormer) etc. open source projects.


## Citation

If you find this code is helpful, please cite our paper

```tex
@article{zhang2023gsmoothface,
  title   = {GSmoothFace: Generalized Smooth Talking Face Generation via Fine Grained 3D Face Guidance},
  author  = {Haiming Zhang and Zhihao Yuan and Chaoda Zheng and Xu Yan and Baoyuan Wang and Guanbin Li and Song Wu and Shuguang Cui and Zhen Li},
  year    = {2023},
  journal = {arXiv preprint arXiv: 2312.07385}
}
```
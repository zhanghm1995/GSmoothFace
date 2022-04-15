## Face3DMMFormer

## Dependencies
Python 3.6
- Check the required python packages in `requirements.txt`.
- ffmpeg

For visualization, you need to install the [nvdiffrast](https://nvlabs.github.io/nvdiffrast/) package, please check the link to compile and install it in your python environment.

## Demo

### Data preparation
First of all, please prepare the driven audio you want to test and place them in `./data/audio_samples` folder.

### Run the demo script
Run the following command in the terminal. And before run the demo, please change the `--checkpoint_dir` in the script to specify the path of saved results.

```bash
bash demo.sh
```

**Note**: if you want to visualize the generated results, you need to specify the `deep3dface_dir` path in the `demo.yaml` file. The visualization results will be save in `work_dir/demo/lightning_logs/version_*/vis/train/*` by default.


### Generate the video
After getting the rendered images, you can generate the video by the following command.

```bash
bash create_video_app.sh
```
And pay attention to the corresponding audio file is placed in the rendered images folder, you can specify this audio path when you run above command to generate the video with audio signal.

## Acknowledgement

We gratefully acknowledge the FaceFormer, Deep3DFace_pytorch, etc. open source projects.


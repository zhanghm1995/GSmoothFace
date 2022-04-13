## Face3DMMFormer

## Environment

- Ubuntu 18.04.1
- Python 3.6 or 3.7
- Pytorch 1.10.0

## Dependencies

- Check the required python packages in `requirements.txt`.
- ffmpeg

## Demo

### Data preparation
First of all, please prepare the driven audio you want to test and place them in `./data/audio_samples` folder.

### Run the demo script
Run the following command in the terminal. And before run the demo, please change the `--checkpoint_dir` in the script to specify the path of saved results.

```bash
bash main_train_one_test.sh
```

### Visualize the results
After the demo, you can see the results in your specified `--checkpoint_dir` folder. And then you can visualize the results by the following command.

```bash
cd visualizer
bash main_render_pred_results.sh
```
Also, you should change the `generated_vertex_path` argument in the script to specify the path of `npy` file of the generated vertex, and other arguments meanings are explained as follows:
```bash
--video_name : the name of the testing video, will rendering the face of this person.
--generated_vertex_path : the path of the generated vertex file.
--output_path : the save path of the output rendering images.
```

### Generate the video
After getting the rendered images, you can generate the video by the following command.

```bash
bash create_video_app.sh
```
And pay attention to the corresponding audio file is placed in the rendered images folder, you can specify this audio path when you run above command to generate the video with audio signal.

## Acknowledgement

We gratefully acknowledge someone.



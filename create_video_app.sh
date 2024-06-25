set -x

name=test_WDA_KimSchrier_000_singing
image_dir="results/demo/test_WDA_KimSchrier_000_singing"
python create_video_app.py --image_root ${image_dir} \
                           --video_fps 25 \
                           --audio_path ${image_dir}/WDA_KimSchrier_000_000.wav \
                           --video_name ${name}
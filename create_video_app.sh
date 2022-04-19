set -x

name=demo
image_dir="work_dir/WDA_KimSchrier_000_000"
python create_video_app.py --image_root ${image_dir} \
                           --video_fps 25 \
                           --audio_path ${image_dir}/audio.wav \
                           --video_name ${name}
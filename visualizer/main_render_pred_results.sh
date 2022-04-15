### You can use this script to render the results of the prediction.
set -x

generated_vertex_path="/home/zhanghm/Research/V100/TalkingFaceFormer/work_dir/demo/lightning_logs/version_7/vis/train/WRA_KellyAyotte_000_000.npy"

python main_render_pred_results.py --data_root ../data/HDTF_face3dmmformer/val \
                                   --video_name WRA_KellyAyotte_000 \
                                   --output_root ../test_dir/demo_audio_sing_song_PPE \
                                   --gen_vertex_path ${generated_vertex_path} \
                                   --need_pose

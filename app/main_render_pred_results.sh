### You can use this script to render the results of the prediction.

generated_vertex_path="work_dir/TVCG/demo_WDA_ChrisMurphy0_000_audio_RD_Radio34_002/lightning_logs/version_0/vis/train/WDA_ChrisMurphy0_000_000.npy"

set -x

python app/main_render_pred_results.py --data_root data/HDTF_face3dmmformer/train \
                                       --video_name WDA_KimSchrier_000 \
                                       --output_root results/demo/WDA_KimSchrier_000_000_audio_RD_Radio34_002 \
                                       --gen_vertex_path ${generated_vertex_path} \
                                       --need_pose
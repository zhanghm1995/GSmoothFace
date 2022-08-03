### You can use this script to render the results of the prediction.

generated_vertex_path="/home/zhanghm/Research/V100/TalkingFaceFormer/work_dir/demo/lightning_logs/version_7/vis/train/WRA_KellyAyotte_000_000.npy"
generated_vertex_path="/home/zhanghm/Research/V100/TalkingFaceFormer/work_dir/AAAI/test/lightning_logs/version_5/vis/0.npy"

set -x
python app/main_render_pred_results.py --data_root data/HDTF_face3dmmformer/train \
                                   --video_name WDA_CarolynMaloney1_000 \
                                   --output_root test_dir/AAAI/test_WDA_CarolynMaloney1_000-Silence_audio \
                                   --gen_vertex_path ${generated_vertex_path} \
                                   --need_pose

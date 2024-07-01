set -x

# checkpoint="work_dir/train_face_3dmm_expression_mouth_mask_paper/lightning_logs/version_4/checkpoints/epoch=167-step=87359.ckpt"
# python demo.py --cfg config/demo.yaml --test_mode \
#                --checkpoint ${checkpoint} \
#                --checkpoint_dir work_dir/test_demo_train


checkpoint="work_dir/TVCG/face3dmm_8_speakers/lightning_logs/version_0/checkpoints/epoch=19-step=10399.ckpt"
python demo.py --cfg config/demo.yaml --test_mode \
               --checkpoint ${checkpoint} \
               --checkpoint_dir work_dir/TVCG/WDA_ChrisMurphy0_000_audio_RD_Radio34_002
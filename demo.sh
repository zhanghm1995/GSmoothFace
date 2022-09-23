set -x

# checkpoint="work_dir/train_face_3dmm_expression_mouth_mask_paper/lightning_logs/version_1/checkpoints_test/epoch=86-step=45239.ckpt"
checkpoint="work_dir/demo_train/lightning_logs/version_0/checkpoints_test/epoch=29-step=15599.ckpt"
checkpoint="work_dir/demo_train/lightning_logs/version_0/checkpoints_test/epoch=14-step=7799.ckpt"
checkpoint="work_dir/demo_train/lightning_logs/version_0/checkpoints_test/epoch=118-step=61879.ckpt"

checkpoint="work_dir/train_face_3dmm_expression_mouth_mask_paper/lightning_logs/version_4/checkpoints/epoch=167-step=87359.ckpt"
python demo.py --cfg config/demo.yaml --test_mode \
               --checkpoint ${checkpoint} \
               --checkpoint_dir work_dir/test_demo_train
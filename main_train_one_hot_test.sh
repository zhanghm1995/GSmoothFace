set -x

checkpoint="work_dir/train_face_3dmm_expression_mouth_mask_paper/lightning_logs/version_4/checkpoints/epoch=167-step=87359.ckpt"
python main_train_one_hot.py --cfg config/face_3dmm_expression_mouth_mask_test.yaml --test_mode \
                             --checkpoint ${checkpoint} \
                             --checkpoint_dir work_dir/train_face_3dmm_expression_mouth_mask_paper/supervise_exp
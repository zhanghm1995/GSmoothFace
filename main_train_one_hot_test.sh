set -x

checkpoint="work_dir/train_face_3dmm_expression_mouth_mask_paper/lightning_logs/version_4/checkpoints/epoch=167-step=87359.ckpt"
checkpoint="work_dir/AAAI/train_one_hot_3dmmformer/lightning_logs/version_3/checkpoints/epoch=19-step=10399.ckpt"

# python main_train_one_hot.py --cfg config/AAAI/face_3dmm_expression_mouth_mask_paper.yaml \
#                              --test_mode \
#                              --checkpoint ${checkpoint} \
#                              --log_dir work_dir/AAAI/test \
                             

checkpoint="work_dir/AAAI/speaker_341_no_template_face/lightning_logs/version_1/checkpoints/epoch=4-step=62719.ckpt"
python main_train_one_hot.py --cfg config/AAAI/speaker_341_no_template_face.yaml \
                             --test_mode \
                             --checkpoint ${checkpoint} \
                             --log_dir work_dir/AAAI/speaker_341_test
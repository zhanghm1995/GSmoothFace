set -x

# python main_train_one_hot.py --cfg config/AAAI/speaker_341_no_template_face.yaml \
#                              --log_dir work_dir/AAAI/speaker_341_no_template_face


python main_train_one_hot.py --cfg config/CVPR/face3dmmformer_wo_speaker_id.yaml \
                             --log_dir work_dir/CVPR/face3dmmformer_wo_speaker_id
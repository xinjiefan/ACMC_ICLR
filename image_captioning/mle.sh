export PATH=$PATH:/home1/06008/xf993/java/bin/
#DISK_PATH=/work/06008/xf993/maverick2/
DISK_PATH=/datadrive/
python train.py \
 --id fc \
 --caption_model fc\
 --input_json data/cocotalk.json\
 --input_fc_dir $DISK_PATH/IC/data/cocotalk_fc \
 --input_att_dir $DISK_PATH/IC/data/cocotalk_att \
 --input_label_h5 data/cocotalk_label.h5\
 --batch_size 10\
 --seq_per_img 5\
 --learning_rate 5e-4\
 --learning_rate_decay_start 0 \
 --scheduled_sampling_start 0 \
 --checkpoint_path $DISK_PATH/IC/model/mle_retrain \
 --save_checkpoint_every 10000\
 --val_images_use 5000 \
 --max_epochs 30 \
 --language_eval 1 \
 --losses_log_every 1000\
 --binary_tree_coding_dir /home/ziyu/self-critical.pytorch\
 --cached_tokens $DISK_PATH/IC/data/coco-train-idxs \
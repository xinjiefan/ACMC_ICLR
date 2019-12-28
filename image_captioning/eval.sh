export PATH=$PATH:/home1/06008/xf993/java/bin/
#DISK_PATH=/work/06008/xf993/maverick2/
DISK_PATH=/datadrive/
python eval.py \
  --dump_images 0 \
  --num_images 5000 \
  --model $DISK_PATH/IC/model/ARSM/5em5_10_/model-best.pth \
  --infos_path $DISK_PATH/IC/model/ARSM/5em5_10_/infos_fc-best.pkl \
  --language_eval 1 \
  --input_fc_dir $DISK_PATH/IC/data/cocotalk_fc\
  --input_att_dir $DISK_PATH/IC/data/cocotalk_att\
  --batch_size 100

#--model $DISK_PATH/IC/model/ARSM/5em5_10/model-best.pth \
#--infos_path $DISK_PATH/IC/model/ARSM/5em5_10/infos_fc-best.pkl \
#--model $DISK_PATH/IC/model/ARSM/5em5_10/model-best.pth \
#--infos_path $DISK_PATH/IC/model/ARSM/5em5_10/infos_fc-best.pkl \




#--input_json data/cocotalk.json\
#--input_fc_dir /work/06008/xf993/maverick2/IC/data/cocotalk_fc\
#--input_att_dir /work/06008/xf993/maverick2/IC/data/cocotalk_att\
#--input_label_h5 data/cocotalk_label.h5 \
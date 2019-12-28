DISK_PATH=/
python train_cmd.py  --signal arm \
              --environment BlackBoxGeneralization \
              --rl_inner_batch 8 \
              --rl_beam 64 \
              --arm_sample sample \
              --decay_factor 1 \
              --logits_factor 0\
              \
              --init_weights $DISK_PATH/exps/supervised_no_grammar/Weights/best.model \
              --nb_epochs 100 \
              --optim_alg Adam \
              --learning_rate 5e-5 \
              --batch_size 16 \
              \
              --train_file $DISK_PATH/data/1m_6ex_karel/train.json \
              --val_file $DISK_PATH/data/1m_6ex_karel/val.json \
              --vocab $DISK_PATH/data/1m_6ex_karel/new_vocab.vocab \
              --result_folder $DISK_PATH/exps/725_arm_16_5em5_sample_nogrammar_trainval\
              \
              \
              --use_cuda \

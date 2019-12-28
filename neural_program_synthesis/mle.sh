DISK_PATH=/
python train_cmd.py --kernel_size 3 \
             --conv_stack "64,64,64" \
             --fc_stack "512" \
             --tgt_embedding_size 256 \
             --lstm_hidden_size 256 \
             --nb_lstm_layers 2 \
             \
             --signal supervised \
             --nb_ios 5 \
             --nb_epochs 100 \
             --optim_alg Adam \
             --batch_size 128 \
             --learning_rate 1e-4 \
             \
             --train_file $DISK_PATH/data/1m_6ex_karel/train.json \
             --val_file $DISK_PATH/data/1m_6ex_karel/val.json \
             --vocab $DISK_PATH/data/1m_6ex_karel/new_vocab.vocab \
             --result_folder $DISK_PATH/exps/supervised_no_grammar_test \
             \
             \
             --use_cuda

python scripts/prepro_feats.py --input_json data/dataset_coco.json --output_dir /work/06008/xf993/maverick2/IC/data/cocotalk --images_root /work/06008/xf993/maverick2/IC/data/image_root

python scripts/make_bu_data.py --output_dir /work/06008/xf993/maverick2/IC/data/cocobu --downloaded_feats /work/06008/xf993/maverick2/IC/data/bu_data

python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl /work/06008/xf993/maverick2/IC/data/coco-train --split train


python3 scripts/prepro_labels.py --input_json /home1/06008/xf993/self-critical.pytorch/data/dataset_coco.json --output_json /home1/06008/xf993/self-critical.pytorch/data/cocotalk.json --output_h5 /home1/06008/xf993/self-critical.pytorch/data/cocotalk --max_length 16




python scripts/prepro_labels.py --input_json /home1/06008/xf993/self-critical.pytorch/data/dataset_coco.json --output_json /home1/06008/xf993/self-critical.pytorch/data/cocotalk.json --output_h5 /home1/06008/xf993/self-critical.pytorch/data/cocotalk --max_length 16

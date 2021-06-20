# example pipeline shell script
# products: data/sku.csv

master_products="data/sku.csv"
products="data/sku.csv"
bert_model="indobenchmark/indobert-base-p1"

# make bert base index for batch selection (create training data from master products)
python -m preprocessing.feature_extraction \
    --products $master_products \
    --model $bert_model \
    --save save/base_index.ann

python -m preprocessing.batch_selection \
    --products $master_products \
    --index save/base_index.ann \
    --output save/matches.csv

python -m preprocessing.serialize \
    --matches save/matches.csv \
    --products $master_products \
    --keep-columns name price \
    --output save/matches.serialized.txt

python -m utils.split \
    --data save/matches.serialized.txt \
    --split 0.8 0.1 0.1 \
    --out-format save/{set}_matches.serialized.txt

python -m susubert.train \
    --train-set save/train_matches.serialized.txt \
    --val-set save/val_matches.serialized.txt \
    --test-set save/test_matches.serialized.txt \
    --save save/susubert


# train s-bert model and get match candidates
python -m blocking.train_blocker \
    --lm $bert_model \
    --data matches.serialized.txt \
    --save save/s-bert # get s-bert model

python -m blocking.blocker \
    --products $master_products \
    --model save/s-bert \
    --save-index save/s-bert.ann \
    --output save/blocked_candidates.csv

python -m preprocessing.serialize \
    --matches save/blocked_candidates.csv \
    --products $master_products \
    --output save/blocked_candidates.serialized.txt

python -m susubert.matcher \
    --model save/susubert \
    --matches save/blocked_candidates.serialized.txt \
    --block-matches save/blocked_candidates.csv \
    --output save/blocked_matches.csv

python -m utils.fin \
    --products $products \
    --matches save/blocked_matches.csv \
    --output save/fin.csv


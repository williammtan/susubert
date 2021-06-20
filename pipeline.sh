# example pipeline shell script
# products: data/sku.csv

master_products="data/sku.csv"
products="data/products.csv"
bert_model="indobenchmark/indobert-base-p1"

# make bert base index for batch selection (create training data from master products)
# python -m preprocessing.feature_extraction --products $master_products --model $bert_model --save save/base_index.ann
python -m preprocessing.batch_selection --products $master_products --index save/base_index.ann --output save/train_matches.csv
python -m preprocessing.serialize --matches save/train_matches.csv --keep-columns name --output save/train_matches.serialized.csv

python -m susubert.train --data save/train_matches.serialized.csv --save save/susubert

# train s-bert model and get match candidates
python -m blocking.train_blocker --base-model $bert_model --save save/s-bert # get s-bert model
python -m blocking.blocker --products $products --output save/blocked_matches.csv
python -m preprocessing.serialize --matches save/blocked_matches.csv --output save/blocked_matches.serialized.csv
python -m susubert.matcher --matches save/blocked_matches.serialized.csv --output save/fin_matches.csv
python -m utils.fin --matches save/fin_matches.csv --output save/fin.csv


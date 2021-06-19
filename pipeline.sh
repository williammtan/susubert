# example pipeline shell script
# products: data/sku.csv

products = "data.sku.csv"
bert_model = "indobenchmark/indobert-base-p1"

# make bert base index for batch selection (create training data from master products)
python preprocessing/feature_extraction --products --model bert_model --output save/base_index.ann
python preprocessing/batch_selection.py --products $products --index save/base_index.ann --output save/train_matches.csv
python preprocessing/serialize.py --matches save/matches.csv --products $products --keep-columns name price --output save/train_matches.serialized.csv

# train s-bert model and get match candidates
python blocking/train_blocker.py --base-model bert_model --save save/s-bert # get s-bert model
python blocking/blocker.py --products $products --output save/blocked_matches.csv
python preprocessing/serialize.py --matches save/blocked_matches.csv --output save/blocked_matches.serialized.csv
python utils/fin.py --matches save/blocked_matches.serialized.csv --output save/fin.csv


import argparse

from ..preprocessing.feature_extraction import make_features_index
from ..preprocessing.serialize import serialize
from ..trainer.utils import make_dataset

from ..utils.gcs import download_from_gcs
from ..utils.utils import get_tokenizer
from ..utils.fin import fin

from ..blocking.train_blocker import train_blocker
from ..blocking.blocker import blocker

from ..susubert.matcher import matcher

import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import tempfile
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lm',
        nargs='+',
        help='Language model',
        default='indobenchmark/indobert-base-p1')
    parser.add_argument(
        '--model',
        nargs='+',
        help='Saved bert model')
    parser.add_argument(
        '--matches',
        nargs='+',
        help='Serialized matches from susubert training',
        default='gs://food-id-app-kubeflowpipelines-default/susubert/sku.csv')
    parser.add_argument(
        '--products',
        nargs='+',
        help='Full products dataset',
        default='gs://food-id-app-kubeflowpipelines-default/susubert/sku.csv')
    parser.add_argument(
        '--job-dir',
        type=str,
        help='GCS or local dir to write checkpoints and export model',
        default='gs://food-id-app-kubeflowpipelines-default/susubert/s-bert')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for s-bert training')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-5,
        help='Learning rate for s-bert training')
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=3,
        help='Number of epochs for s-bert training')
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Top k matches for blocker')
    parser.add_argument(
        '--threshold',
        type=int,
        required=False,
        help='Cosine similarity threshold for blocker')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    DATA_DIR = os.path.join(tempfile.gettempdir(), 'susubert-matcher')
    tf.io.gfile.makedirs(DATA_DIR)

    matches_fname = os.path.join(DATA_DIR, 'matches.csv')
    products_fname = os.path.join(DATA_DIR, 'products.csv')
    index_fname = os.path.join(DATA_DIR, 'sbert-index.ann')
    model_dir = os.path.join(DATA_DIR, 'susubert/')
    matches_fname = os.path.join(DATA_DIR, 'block_matches_result.jsonlines')
    master_products_fname = os.path.join(DATA_DIR, 'master_products.csv')

    download_from_gcs(gcs_url=args.matches, fname=matches_fname)
    download_from_gcs(gcs_url=args.matches, fname=matches_fname)
    download_from_gcs(gcs_url=args.model, fname=model_dir)

    matches = pd.read_csv(matches_fname)
    products = pd.read_csv(products_fname)

    # train blocker
    sbert = train_blocker(args.lm, matches)
    tokenizer = get_tokenizer(args.lm)

    match_df = blocker(sbert, products, index_fname, args.top_k, args.threshold)
    serialized_match_df = serialize(match_df, products, keep_columns=['name', 'price'])

    match_dataset = make_dataset(serialized_match_df, tokenizer)

    matcher(args.lm, model_dir, serialized_match_df, match_df, args.batch_size, matches_fname)

    master_products = fin(products, matches_fname)
    master_products.to_csv(master_products_fname)




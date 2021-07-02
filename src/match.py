import argparse

from .preprocessing.feature_extraction import make_features_index
from .preprocessing.serialize import serialize

from .utils.utils import make_dataset
from .utils.gcs import download_from_gcs, download_directory_from_gcs, upload_blob
from .utils.utils import get_tokenizer
from .utils.fin import fin

from .blocking.train_blocker import train_blocker
from .blocking.blocker import blocker

from .susubert.matcher import matcher

from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import tempfile
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lm',
        help='Language model',
        default='indobenchmark/indobert-base-p1')
    parser.add_argument(
        '--model',
        default='gs://food-id-app-kubeflowpipelines-default/susubert/susubert-model/',
        help='Saved bert model')
    parser.add_argument(
        '--matches',
        help='Serialized matches from susubert training',
        default='gs://food-id-app-kubeflowpipelines-default/susubert/matches.serialized.csv')
    parser.add_argument(
        '--products',
        help='Full products dataset',
        default='gs://food-id-app-kubeflowpipelines-default/susubert/sku.csv')
    parser.add_argument(
        '--col-features',
        nargs='+',
        help='Features used for bert training',
        default=['name', 'price'])
    parser.add_argument(
        '--job-dir',
        type=str,
        help='GCS or local dir to write checkpoints and export model',
        default='gs://food-id-app-kubeflowpipelines-default/susubert/s-bert')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
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
        # default=0.4,
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
    matches_result_fname = os.path.join(DATA_DIR, 'matches_result.csv')
    master_products_fname = os.path.join(DATA_DIR, 'fin.csv')

    download_from_gcs(gcs_url=args.matches, fname=matches_fname)
    download_from_gcs(gcs_url=args.products, fname=products_fname)
    download_directory_from_gcs(args.model, model_dir)

    matches = pd.read_csv(matches_fname)
    products = pd.read_csv(products_fname)
    print(matches)

    # train blocker
    sbert = train_blocker(args.lm, matches)
    tokenizer = get_tokenizer(args.lm)

    match_df = blocker(sbert, products, index_fname, args.top_k, args.threshold)

    serialized_match_df = serialize(match_df, products, keep_columns=args.col_features)


    #match_dataset = make_dataset(serialized_match_df, tokenizer)

    match_results = matcher(args.lm, model_dir, serialized_match_df, match_df, args.batch_size)

    master_products = fin(products, match_results)
    match_results.to_csv(matches_result_fname)
    master_products.to_csv(master_products_fname)
    upload_blob(os.path.join(args.job_dir, 'matches_result.csv'), matches_result_fname)
    upload_blob(os.path.join(args.job_dir, 'fin.csv'), master_products_fname)


# gcloud ai-platform local train --module-name src.match --job-dir gs://ml_foodid_project/product-matching/susubert/  --package-path src -- --products gs://ml_foodid_project/product-matching/susubert/000000000004.csv --matches gs://ml_foodid_project/product-matching/susubert/matches.serialized.csv

import argparse

from .preprocessing.feature_extraction import make_features_index
from .preprocessing.batch_selection import batch_selection
from .preprocessing.serialize import serialize
from .utils.utils import make_dataset
from .utils.gcs import download_from_gcs, upload_blob, upload_local_directory_to_gcs
from .susubert.train import create_bert_model

import tensorflow as tf
from tensorflow.distribute import MirroredStrategy # pyright: reportMissingImports=false
from sklearn.model_selection import train_test_split
import os
import tempfile
import pandas as pd

strategy = MirroredStrategy()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lm',
        help='Language model',
        default='indobenchmark/indobert-base-p1')
    parser.add_argument(
        '--products',
        help='Full products csv file',
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
        default='gs://food-id-app-kubeflowpipelines-default/susubert/')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-5,
        help='Learning rate for Adam')
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=3,
        help='Number of epochs for training')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    DATA_DIR = os.path.join(tempfile.gettempdir(), 'susubert')
    tf.io.gfile.makedirs(DATA_DIR)

    products_fname = os.path.join(DATA_DIR, 'products.csv')
    serialized_match_fname = os.path.join(DATA_DIR, 'match.serialized.csv')
    model_dir = os.path.join(DATA_DIR, 'susubert-model/')

    download_from_gcs(gcs_url=args.products, fname=products_fname)

    products = pd.read_csv(products_fname)
    index = make_features_index(args.lm, products)

    match_df = batch_selection(products, index)
    serialized_match_df = serialize(match_df, products, keep_columns=args.col_features)
    serialized_match_df.to_csv(serialized_match_fname)
    upload_blob(os.path.join(args.job_dir, 'matches.serialized.csv'), serialized_match_fname)

    train, test = train_test_split(serialized_match_df, test_size=0.2)
    test, val = train_test_split(test, test_size=0.5)
    
    with strategy.scope():
        model, tokenizer = create_bert_model(args.lm, args.learning_rate)

    print(f"Train set count: {len(train)}")
    print(f"Val set count: {len(val)}")
    print(f"Test set count: {len(test)}")
    train_dataset = make_dataset(train, tokenizer)
    val_dataset = make_dataset(val, tokenizer)
    test_dataset = make_dataset(test, tokenizer)

    model.fit(
        train_dataset.shuffle(len(train_dataset)).batch(args.batch_size), epochs=args.n_epochs, 
        batch_size=args.batch_size, validation_data=val_dataset.shuffle(len(val_dataset)).batch(args.batch_size),
    )

    print(f"<============= Test evaluation =============>")
    model.evaluate(test_dataset.batch(args.batch_size), return_dict=True, batch_size=args.batch_size)

    model.save_pretrained(model_dir)

    upload_local_directory_to_gcs(model_dir, os.path.join(args.job_dir, 'susubert-model/'))


import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, TFAutoModel
import tensorflow as tf
import os

from utils.index import make_index
from utils.gcs import download_from_gcs
import tempfile

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, axis=-1), token_embeddings.shape)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=np.inf)
    return sum_embeddings / sum_mask


def feature_extraction(model, tokenizer, sentences, batch_size=1000):
    def get_embeddings(sentences):
        encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128)
        dataset = tf.data.Dataset.from_tensor_slices((
            dict(encoded_input),
        ))

        model_output = model.predict(dataset.batch(32))

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    embedding = np.empty((0,768), int)
    for names in tqdm(batch(sentences, 1000), total = len(sentences.name) // batch_size):
        embedding = np.append(embedding, get_embeddings(names.tolist()), axis=0)
    
    return embedding
        
def make_features_index(model, products):
    tokenizer = BertTokenizer.from_pretrained(model)
    model = TFAutoModel.from_pretrained(model)

    feature_embedding= feature_extraction(model, tokenizer, products.name)

    index, _ = make_index(feature_embedding, products.id.values)
    return index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bucket')
    parser.add_argument('-p', '--products')
    parser.add_argument('-m', '--model')
    parser.add_argument('-s', '--save')
    args = parser.parse_args()

    tempdir = tempfile.mkdtemp(prefix='feature-extractor')
    products_temp = os.path.join(tempdir, 'products.csv')
    index_temp = os.path.join(tempdir, 'index.csv')

    download_blob(args.bucket, args.products, )
    products = pd.read_csv(args.products)

    index = make_features_index(args.model, products, products_temp)
    index.save(os.path.join(tempdir, index_temp))
    upload_to_storage(args.save, index_temp)



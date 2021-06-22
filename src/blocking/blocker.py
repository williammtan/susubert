from ..utils.index import make_index, load_index
from ..preprocessing.serialize import serialize_products
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import argparse
import os

def index_sbert(sbert, products):
    sents = serialize_products(products, features=['name'])
    embeddings = sbert.encode(sents, convert_to_numpy=True)
    index, ids_mapping = make_index(embeddings, products.id)

    return index

def blocker(args):
    products = pd.read_csv(args.products)

    sbert = SentenceTransformer(args.model)
    if not os.path.isfile(args.save_index):
        index = index_sbert(sbert, products)
        index.save(args.save_index)
    else:
        index = load_index(args.save_index, dimensions=768)

    match_candidates = []

    for i, prod in products.iterrows():
        nn = np.array(index.get_nns_by_item(i, n=args.top_k, include_distances=True)).T
        if args.threshold:
            nn = nn[[nn[:, 1] > args.threshold]]
        nn_ids = products[products.index.isin(nn[:, 0])].id.values
        for id in nn_ids:
            match_candidates.append((prod.id, id))
    
    match_candidates = np.array(match_candidates, )
    match_candidates.sort()
    match_candidates = np.unique(match_candidates, axis=0)
    
    output = open(args.output, 'w')
    output.write('id1,id2\n')
    for match in match_candidates:
        output.write(f'{match[0]}, {match[1]} \n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--products')
    parser.add_argument('--model')
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--save-index')
    parser.add_argument('--output')
    args = parser.parse_args()

    blocker(args)
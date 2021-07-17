from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import math
from pathlib import Path
from annoy import AnnoyIndex

from sentence_transformers import SentenceTransformer



def index_sbert(sbert, sents):
    embeddings = sbert.encode(sents, convert_to_numpy=True)
    index = AnnoyIndex(768)
    for i, vec in tqdm(enumerate(embeddings)):
        index.add_item(i, vec)
    index.build(10)

    return index

def blocker(sbert, products, args):
    index = index_sbert(sbert, products.name.values)
    only_masters = products[products.master_product.notnull()]
    master_products = only_masters.master_product.unique()

    candid_matches = []
    for i, prod in tqdm(products.iterrows()):
        nn = index.get_nns_by_item(i, 100)
        nn = [n for n in nn if n in only_masters.index][:args.top_k]
        for idx in nn:
            mp = products.iloc[idx]
            candid_matches.append({
                "id1": prod.id,
                "id2": mp.master_product_id,
                "sent1": prod['name'],
                "sent2": mp.master_product,
            })
    candid_matches = pd.DataFrame(candid_matches).drop_duplicates(subset=['id1', 'id2'])
    candid_matches = candid_matches.dropna(subset=['sent1', 'sent2'])
    return candid_matches

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--products')
    parser.add_argument('--s-bert')
    parser.add_argument('--top-k', type=int)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--save-matches')
    args = parser.parse_args()

    products = pd.read_csv(args.products)

    sbert = SentenceTransformer(args.s_bert)
    matches = blocker(sbert, products, args)

    Path(args.save_matches).parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.save_matches, index=False)


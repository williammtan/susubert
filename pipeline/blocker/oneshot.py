from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import math
from pathlib import Path
from faiss import IndexFlatL2

from sentence_transformers import SentenceTransformer



def index_sbert(sbert, sents):
    embeddings = sbert.encode(sents, convert_to_numpy=True)
    index = IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype(np.float32))

    return index

def blocker(sbert, products, master_products, args):
    index = index_sbert(sbert, master_products.name.values)
    product_embeddings = sbert.encode(products.name.values, convert_to_numpy=True)
    only_masters = products[products.master_product.notnull()]

    candid_matches = []
    for i, prod in tqdm(products.iterrows()):
        product_vec = product_embeddings[[i]]
        distances, nn = index.search(product_vec, args.top_k)
        nn = [mp for d, mp in zip(distances[0], nn[0]) if d > args.threshold]
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
    parser.add_argument('--master-products')
    parser.add_argument('--s-bert')
    parser.add_argument('--top-k', type=int)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--save-matches')
    args = parser.parse_args()

    products = pd.read_csv(args.products)
    master_products = pd.read_csv(args.master_products)

    sbert = SentenceTransformer(args.s_bert)
    matches = blocker(sbert, products, master_products, args)

    Path(args.save_matches).parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.save_matches, index=False)


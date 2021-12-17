from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import math
from pathlib import Path
import faiss

from sentence_transformers import SentenceTransformer



def index_sbert(sbert, sents):
    embeddings = sbert.encode(sents, convert_to_numpy=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    embeddings = np.array(embeddings).astype(np.float32)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    return index

def blocker(sbert, products, master_products, args):
    index = index_sbert(sbert, master_products.name.values)
    product_embeddings = sbert.encode(products.name.values, convert_to_numpy=True)
    faiss.normalize_L2(product_embeddings)

    candid_matches = []
    for i, prod in tqdm(products.iterrows()):
        product_vec = product_embeddings[[i]]
        distances, nn = index.search(product_vec, args.top_k)
        nn = [mp for d, mp in zip(distances[0], nn[0]) if d > args.threshold]
        for idx in nn:
            mp = master_products.iloc[idx]
            candid_matches.append({
                "id1": prod.id,
                "id2": mp.id,
                "sent1": prod['name'],
                "sent2": mp['name'],
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


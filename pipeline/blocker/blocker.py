from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import math
from pathlib import Path
from annoy import AnnoyIndex

from sentence_transformers import SentenceTransformer

EITHER_COLUMNS = ['volume', 'weight']
SAME_COLUMNS = ['quantity']

def index_sbert(sbert, sents):
    embeddings = sbert.encode(sents, convert_to_numpy=True)
    index = AnnoyIndex(768)
    for i, vec in tqdm(enumerate(embeddings)):
        index.add_item(i, vec)
    index.build(10)

    return index


def blocker(sbert, products, args):
    def get_same_prods(target_prod):
        same_prods = products.copy()
        either = None
        for col in EITHER_COLUMNS:
            if target_prod[col] == type(float) and math.isnan(target_prod[col]):
                continue
            if either is None:
                either = same_prods[col] == target_prod[col]
            else:
                either = np.logical_or(either, same_prods[col] == target_prod[col])
        same_prods = same_prods[either]

        same = None
        for col in SAME_COLUMNS:
            if same is None:
                same = same_prods[col] == target_prod[col]
            else:
                same = np.logical_and(same, same_prods[col] == target_prod[col])
        same_prods = same_prods[same]

        # if 'master_product' in products.columns:
        #     if target_prod.master_product is not None:
        #         # either the master product is None or the master products are equal
        #         either = np.logical_or(same_prods.master_product.isnull(), same_prods.master_product == target_prod.master_product)
        #         same_prods = same_prods[either]

        return same_prods


    index = index_sbert(sbert, products.sent.values)

    match_candidates = []

    for i, prod in tqdm(products.iterrows(), total=len(products)):
        same_prods = get_same_prods(prod)
        nn = np.array(index.get_nns_by_item(i, n=args.top_k, include_distances=True)).T
        if args.threshold:
            nn = nn[([nn[:, 1] < (2-(args.threshold*2))])]
        nn_ids = same_prods[same_prods.index.isin(nn[:, 0])].id.values
        nn_ids = [id for id in nn_ids if id != prod.id]
        for id in nn_ids:
            match_candidates.append((prod.id, id))
    
    match_candidates = np.array(match_candidates)
    match_candidates.sort(axis=1)

    match_candidates = np.unique(match_candidates, axis=1).T

    return pd.DataFrame({'id1': match_candidates[0], 'id2': match_candidates[1] })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--products')
    parser.add_argument('--serialized-products')
    parser.add_argument('--s-bert')
    parser.add_argument('--top-k', type=int)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--save-matches')
    args = parser.parse_args()

    products = pd.read_csv(args.products)
    serialized_products = pd.read_csv(args.serialized_products)
    products = products.merge(serialized_products, on='id')

    sbert = SentenceTransformer(args.s_bert)
    matches = blocker(sbert, products, args)

    Path(args.save_matches).parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.save_matches, index=False)

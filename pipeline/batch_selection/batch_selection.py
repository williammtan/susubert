import argparse
import pandas as pd
import numpy as np
from itertools import combinations
from annoy import AnnoyIndex
from pathlib import Path
from tqdm import tqdm


def negative_hard(products, index):
    """Selects most similar products with different master products"""
    master_products = products.master_product.unique()

    for master in tqdm(master_products):
        offers = products[products.master_product == master] # offers in the master product
        search_size = round(len(offers) * 1.5)
        for i in offers.index:
            top_n = index.get_nns_by_item(i, search_size, include_distances=True)
            top_n = [top_n[0][i] for i, dis in enumerate(top_n[1])]
            non_master = [products.iloc[o].id for o in top_n if products.iloc[o].master_product != master]
            for non in non_master:
                yield (non, offers.loc[i].id)

def positive_all(products):
    id_combinations = np.array(list(combinations(products.id.values, 2)))
    for id1, id2 in tqdm(id_combinations):
        match = 1 if products[products.id == id1].master_product.iloc[0] == products[products.id == id2].master_product.iloc[0] else 0
        if match:
            yield (id1, id2)

def batch_selection(products, index):
    def make_df(matches, label):
        matches.sort()
        matches = np.unique(matches, axis=1)
        return pd.DataFrame({'id1': matches[:, 0], 'id2': matches[:, 1], 'match': [label] * len(matches)})

    neg_matches = np.array(list(negative_hard(products, index)))
    pos_matches = np.array(list(positive_all(products)))

    matches = pd.concat([make_df(neg_matches, 0), make_df(pos_matches, 1)])
    return matches

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--products')
    parser.add_argument('--index')
    parser.add_argument('--save-matches')
    args = parser.parse_args()

    index = AnnoyIndex(768)
    index.load(args.index)

    products = pd.read_csv(args.products)
    matches = batch_selection(products, index)

    Path(args.save_matches).parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.save_matches, index=False)

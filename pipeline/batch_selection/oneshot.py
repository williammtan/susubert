import argparse
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from pathlib import Path
from tqdm import tqdm

def batch_selection(products, index):
    matches = []
    for i, prod in tqdm(products.iterrows()):
        nn = index.get_nns_by_item(i, 10)
        nn = [n for n in nn if products[products.index == n].iloc[0].master_product != prod.master_product]
        for idx in nn:
            matches.append({
                "sent1": prod['name'],
                "sent2": products.iloc[idx].master_product,
                "match": 0
            })
        matches.append({
            "sent1": prod['name'],
            "sent2": prod.master_product,
            "match": 1
        })
    matches = pd.DataFrame(matches)

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

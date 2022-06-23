import argparse
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from pathlib import Path
from tqdm import tqdm


def batch_selection(master_products, products, product_index, master_product_index):
    matches = []

    # create all the negative samples, by iterating and searching the nearest products to a master product which are NOT in the master product
    for i, mp in master_products.iterrows():
        vector = master_product_index.get_item_vector(i)
        similar_products = product_index.get_nns_by_vector(vector, 10)
        for idx in similar_products:
            matches.append({
                "sent1": products.iloc[idx],
                "sent2": mp['name'],
                "match": 0
            })

        # generate all positive samples
        labelled_products = products[products.master_product_id==mp.id]
        for p_name in labelled_products.name:
            matches.append({
                "sent1": p_name,
                "sent2": mp['name'],
                "match": 1
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--products')
    parser.add_argument('--master-products')
    parser.add_argument('--product-index')
    parser.add_argument('--master-product-index')
    parser.add_argument('--save-matches')
    args = parser.parse_args()

    product_index = AnnoyIndex(768)
    product_index.load(args.product_index)

    master_product_index = AnnoyIndex(768)
    master_product_index.load(args.master_product_index)

    products = pd.read_csv(args.products)
    master_products = pd.read_csv(args.master_products)
    matches = batch_selection(master_products, products, product_index, master_product_index)

    Path(args.save_matches).parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.save_matches, index=False)

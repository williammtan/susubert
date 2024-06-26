################################################################################
### This script creates match/not match data from master products
### 
### Expected schema:
###     id
###     name
###     description
###     price
###     weight
###     master_product
###
################################################################################

import argparse
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm

from utils.index import load_index

def negative_hard(products, index):
    """Selects most similar products with different master products"""
    master_products = products.master_product.unique()

    for master in master_products:
        offers = products[products.master_product == master] # offers in the master product
        search_size = round(len(offers) * 1.5)
        for i in offers.index:
            top_n = index.get_nns_by_item(i, search_size, include_distances=True)
            top_n = [top_n[0][i] for i, dis in enumerate(top_n[1])]
            non_master = [products.iloc[o].id for o in top_n if products.iloc[o].master_product != master]
            for non in non_master:
                yield (non, offers.loc[i].id)


def negative_master_product(products):
    """Selects most similar master products and the most similar products in the other master product"""
    pass


def positive_random(products):
    """Selects random combinations of offers in master product"""
    master_products = products.master_product.unique()

    for master in master_products:
        offers = products[products.master_product == master] # offers in the master product
        offer_pairs = [np.random.choice(offers.id.values, 2, replace=False) for i in range(100)]
        for pair in offer_pairs:
            yield pair
    

def positive_hard(products, index):
    """Selects most different offers in a master product"""
    master_products = products.master_product.unique()

    for master in master_products:
        offers = products[products.master_product == master] # offers in the master product
        for target_offer in offers.index:
            most_different = np.argmax([index.get_distance(target_offer, o) for o in offers.index])
            yield (offers.loc[target_offer].id, offers.loc[offers.index[most_different]].id)
    
def positive_all(products):
    id_combinations = np.array(list(combinations(products.id.values, 2)))
    match_df = []
    for id1, id2 in tqdm(id_combinations):
        match = 1 if products[products.id == id1].master_product.iloc[0] == products[products.id == id2].master_product.iloc[0] else 0
        match_df.append({
            "id1":id1,
            "id2":id2,
            "match":match
        })
    match_df = pd.DataFrame(match_df)
    
    return match_df[match_df.match == 1]
        
def batch_selection(products, index):
    
    match_df = []
    negative_pairs = list(negative_hard(products, index))
    for neg in negative_pairs:
        name1, name2 = products[products.id == neg[0]].name.iloc[0], products[products.id == neg[1]].name.iloc[0]
        match_df.append({
            "id1": neg[0],
            "name1": name1,
            "name2": name2,
            "id2": neg[1],
            "match": 0
        })
    
    positive_pairs = list(positive_hard(products, index))
    for pos in positive_pairs:
        name1, name2 = products[products.id == pos[0]].name.iloc[0], products[products.id == pos[1]].name.iloc[0]
        match_df.append({
            "id1": pos[0],
            "name1": name1,
            "name2": name2,
            "id2": pos[1],
            "match": 1
        })
    
    match_df = pd.concat([pd.DataFrame(match_df), positive_all(products)])
    
    match_df = match_df.drop_duplicates(subset=['id1', 'id2'])
    return match_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--products')
    parser.add_argument('-i', '--index')
    parser.add_argument('-d', '--index-dims', default=768)
    parser.add_argument('-o', '--output')
    
    args = parser.parse_args()

    match_df = batch_selection(args.products, args.index, args.index_dims)
    match_df.to_csv(args.output, index=False)


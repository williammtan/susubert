from utils.index import make_index, load_index
from preprocessing.serialize import serialize_products
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import argparse
import os

def index_sbert(sbert, products):
    sents = serialize_products(products, features=['name'])
    embeddings = sbert.encode(sents, convert_to_numpy=True)
    # embeddings = np.append(embeddings, (products.price.values / 1000).reshape(len(products), 1), axis=1)
    # tfidf_vectorizer = TfidfVectorizer()
    # embeddings = tfidf_vectorizer.fit_transform(products.name)
    index, ids_mapping = make_index(embeddings, products.id)

    return index

def dims_equal(dim1, dim2):
    if dim1 != np.nan and dim2 != np.nan:
        return dim1 == dim2
    else:
        return False

def blocker(sbert, products, save_index, top_k, threshold, either_columns=['volume', 'weight'], same_columns=['quantity'], overide=True):
    def get_same_prods(target_prod):
        same_prods = products.copy()
        either = None
        for col in either_columns:
            if target_prod[col] == type(float) and math.isnan(target_prod[col]):
                continue
            if either is None:
                either = same_prods[col] == target_prod[col]
            else:
                either = np.logical_or(either, same_prods[col] == target_prod[col])
        same_prods = same_prods[either]

        same = None
        for col in same_columns:
            if same is None:
                same = same_prods[col] == target_prod[col]
            else:
                same = np.logical_and(same, same_prods[col] == target_prod[col])
        same_prods = same_prods[same]

        return same_prods


    if not os.path.isfile(save_index) or overide:
        index = index_sbert(sbert, products)
        index.save(save_index)
    else:
        print('loading_index')
        index = load_index(save_index, dimensions=768+1)

    match_candidates = []

    for i, prod in tqdm(products.iterrows(), total=len(products), position=0, leave=True):
        same_prods = get_same_prods(prod)
        nn = np.array(index.get_nns_by_item(i, n=top_k, include_distances=True)).T
        if threshold:
            nn = nn[([nn[:, 1] < (2-(threshold*2))])]
        nn_ids = same_prods[same_prods.index.isin(nn[:, 0])].id.values
        nn_ids = [id for id in nn_ids if id != prod.id]
        # nn_ids = [id for id in nn_ids if np.any([dims_equal(products[products.id == id][dim].iloc[0], prod[dim]) for dim in either_columns]) and np.all([dims_equal(products[products.id == id][dim].iloc[0], prod[dim]) for dim in same_columns])]
        # if len(nn_ids) == 0:
        #     print(prod)
        for id in nn_ids:
            match_candidates.append((prod.id, id))
    
    match_candidates = np.array(match_candidates, )
    match_candidates.sort(axis=1)

    match_candidates = np.unique(match_candidates, axis=1).T

    return pd.DataFrame({'id1': match_candidates[0], 'id2': match_candidates[1] })
    
    # output = open(args.output, 'w')
    # output.write('id1,id2\n')
    # for match in match_candidates:
    #     output.write(f'{match[0]}, {match[1]} \n')


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
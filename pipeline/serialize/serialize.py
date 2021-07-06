# convert {'id1': 1, 'id2': 2, 'match': 0} -> 
# [CLS] [COL] title [VAL] susu indomilk 200ml [COL] price [VAL] 13000
# [SEP] [COL] title [VAL] susu beruang 200ml [COL] price [VAL] 15000 [SEP]
# , 0

import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path
import json

def serialize_products(products, features):
    serialized = []
    for i, match in tqdm(products.iterrows(), total=len(products)):
        serialized_matches = ' '.join(['COL ' + col + ' VAL ' + str(match[col]) for col in features])

        serialized.append(serialized_matches)
    
    return serialized

def serialize_matches(matches, features):
    serialized = []
    for i, match in tqdm(matches.iterrows(), total=len(matches)):
        serialized_matches = [' '.join(['COL ' + col + ' VAL ' + str(match[col+str(i)]) for col in features]) for i in range(1,3)]

        serialized.append(serialized_matches)
    
    return serialized

def serialize_product_matches(matches, products, features):
    """Serialize matches given a products matches"""
    serialized = [[' '.join(['COL ' + col + ' VAL ' + str(products[products.id == match['id'+str(i)]][col].iloc[0]) for col in features]) for i in range(1,3)] for _, match in tqdm(matches.iterrows(), total=len(matches))]
    
    return serialized

def serialize(matches=None, products=None, keep_columns=['name', 'price']):
    match_df = []

    if matches is not None and 'match' in matches.columns:
        if products is not None:
            sent_pairs = serialize_product_matches(matches, products, features=keep_columns)
        else:
            sent_pairs = serialize_matches(matches, features=keep_columns)

        for match_pair, is_match in zip(sent_pairs, matches.match.values):
            match_df.append({
                "sent1": match_pair[0],
                "sent2": match_pair[1],
                "match": is_match
            })

    elif products is not None:
        if matches is not None and 'id1' in matches.columns:
            sent_pairs = serialize_product_matches(matches, products, features=keep_columns)
            for sent1, sent2 in sent_pairs:
                match_df.append({
                    "sent1": sent1,
                    "sent2": sent2,
                })
            
        else:
            sent_pairs = serialize_products(products, features=keep_columns)

            for serialized_sent, id in zip(sent_pairs, products.id):
                match_df.append({
                    "id": id,
                    "sent": serialized_sent,
                })
    else:
        sent_pairs = serialize_matches(matches, features=keep_columns)

        for sent1, sent2 in sent_pairs:
            match_df.append({
                "sent1": sent1,
                "sent2": sent2,
            })


    match_df = pd.DataFrame(match_df)
    return match_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matches', required=True)
    parser.add_argument('--products', required=True)
    parser.add_argument('--keep-columns')
    parser.add_argument('--save-matches')
    args = parser.parse_args()

    try:
        matches = pd.read_csv(args.matches)
    except pd.errors.EmptyDataError:
        matches = None
    
    try:
        products = pd.read_csv(args.products)
    except pd.errors.EmptyDataError:
        products = None

    serialized_matches = serialize(matches, products, json.loads(args.keep_columns))

    Path(args.save_matches).parent.mkdir(parents=True, exist_ok=True)
    serialized_matches.to_csv(args.save_matches, index=False)

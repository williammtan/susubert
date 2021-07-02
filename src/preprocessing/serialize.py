# convert {'id1': 1, 'id2': 2, 'match': 0} -> 
# [CLS] [COL] title [VAL] susu indomilk 200ml [COL] price [VAL] 13000
# [SEP] [COL] title [VAL] susu beruang 200ml [COL] price [VAL] 15000 [SEP]
# , 0

import pandas as pd
import argparse
from tqdm import tqdm

def serialize_products(products, features):
    serialized = []
    for i, match in tqdm(products.iterrows()):
        serialized_matches = ' '.join(['COL ' + col + ' VAL ' + str(match[col]) for col in features])

        serialized.append(serialized_matches)
    
    return serialized

def serialize_matches(matches, features):
    serialized = []
    for i, match in tqdm(matches.iterrows()):
        serialized_matches = [' '.join(['COL ' + col + ' VAL ' + match[col+str(i)] for col in features]) for i in range(1,3)]

        serialized.append(serialized_matches)
    
    return serialized

def serialize_product_matches(matches, products, features):
    """Serialize matches given a products matches"""
    serialized = [[' '.join(['COL ' + col + ' VAL ' + str(products[products.id == match['id'+str(i)]][col].iloc[0]) for col in features]) for i in range(1,3)] for _, match in tqdm(matches.iterrows(), total=len(matches))]
    # for i, match in tqdm(matches.iterrows()):
    #     serialized_matches = [' '.join(['COL ' + col + ' VAL ' + str(products[products.id == match['id'+str(i)]][col].iloc[0]) for col in features]) for i in range(1,3)]
        # serialized_matches = [products[products.id == match['id'+str(i)]]['name'].iloc[0] for i in range(1,3)]

        # serialized.append(serialized_matches)
    
    return serialized

def serialize(matches, products, keep_columns):
    match_df = []

    if 'match' in matches.columns:
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
        if 'id1' in matches.columns:
            sent_pairs = serialize_product_matches(matches, products, features=keep_columns)
            for sent1, sent2 in sent_pairs:
                match_df.append({
                    "sent1": sent1,
                    "sent2": sent2,
                })
            
        else:
            sent_pairs = serialize_products(matches, features=keep_columns)

            for serialized_sent, id in zip(sent_pairs, matches.id):
                match_df.append({
                    "id": id,
                    "sent": serialized_sent,
                })
    match_df = pd.DataFrame(match_df)
    return match_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matches')
    parser.add_argument('--products')
    parser.add_argument('--keep-columns', nargs='+', default=['name'])
    parser.add_argument('--output')
    args = parser.parse_args()

    serialize(args)

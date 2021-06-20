# convert {'id1': 1, 'id2': 2, 'match': 0} -> 
# [CLS] [COL] title [VAL] susu indomilk 200ml [COL] price [VAL] 13000
# [SEP] [COL] title [VAL] susu beruang 200ml [COL] price [VAL] 15000 [SEP]
# , 0

import pandas as pd
import argparse

def serialize_products(products, features):
    serialized = []
    for i, match in products.iterrows():
        serialized_matches = ' '.join(['COL ' + col + ' VAL ' + match[col] for col in features])

        serialized.append(serialized_matches)
    
    return serialized

def serialize_matches(matches, features):
    serialized = []
    for i, match in matches.iterrows():
        serialized_matches = [' '.join(['COL ' + col + ' VAL ' + match[col+str(i)] for col in features]) for i in range(1,3)]

        serialized.append(serialized_matches)
    
    return serialized

def serialize_product_matches(matches, products, features):
    """Serialize matches given a products matches"""
    serialized = []
    for i, match in matches.iterrows():
        serialized_matches = [' '.join(['COL ' + col + ' VAL ' + products[products.id == match['id'+str(i)]][col].iloc[0] for col in features]) for i in range(1,3)]

        serialized.append(serialized_matches)
    
    return serialized

def serialize(args):
    matches = pd.read_csv(args.matches)
    f = open(args.output, 'w')

    if 'match' in matches.columns:
        sent_pairs = serialize_matches(matches, features=args.keep_columns)
        for match_pair, is_match in zip(sent_pairs, matches.match.values):
            f.write(match_pair[0] + '\t' + match_pair[1] + '\t' + str(is_match) + '\n')

    elif 'id1' in matches.columns and 'match' not in matches.columns and args.products:
        products = pd.read_csv(args.products)
        sent_pairs = serialize_product_matches(matches, products, features=args.keep_columns)
        for (sent1, sent2) in sent_pairs:
            f.write(sent1 + '\t' + sent2 + '\n')

    elif args.products:
        sents = serialize_products(matches, features=args.keep_columns)
        for serialized_sent, id in zip(sents, matches.id):
            f.write(id + '\t' + serialized_sent + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matches')
    parser.add_argument('--products')
    parser.add_argument('--keep-columns', nargs='+', default=['name'])
    parser.add_argument('--output')
    args = parser.parse_args()

    serialize(args)

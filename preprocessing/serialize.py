# convert {'id1': 1, 'id2': 2, 'match': 0} -> 
# [CLS] [COL] title [VAL] susu indomilk 200ml [COL] price [VAL] 13000
# [SEP] [COL] title [VAL] susu beruang 200ml [COL] price [VAL] 15000 [SEP]
# , 0

import pandas as pd
import argparse

def serialize(args):
    matches = pd.read_csv(args.matches)

    serialized = []
    for i, match in matches.iterrows():
        serialized_matches = [' '.join(['COL ' + col + ' VAL ' + match[col+str(i)] for col in args.keep_columns]) for i in range(1,3)]

        serialized.append(serialized_matches)
    
    f = open(args.output, 'w')
    for (match_str), is_match in zip(serialized, matches.match.values):
        f.write(match_str[0] + '\t' + match_str[1] + '\t' + str(is_match) + '\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matches')
    parser.add_argument('--keep-columns', nargs='+', default=['name'])
    parser.add_argument('--output')
    args = parser.parse_args()

    serialize(args)

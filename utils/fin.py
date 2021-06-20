import argparse
import pandas as pd
from igraph import Graph, plot

def fin(args):
    products = pd.read_csv(args.products)
    id_mapping = dict(zip(products.id, range(len(products))))
    matches = pd.read_csv(args.matches)

    g = Graph()
    g.add_vertices(len(products))
    g.add_edges([[id_mapping[id] for id in match[['id1', 'id1']].tolist()] for _, match in matches.iterrows() if match.match == 1])
    clusters = list(g.clusters())

    for i, c in enumerate(clusters):
        for prod_idx in c:
            products.loc[prod_idx]['fin'] = i
    
    products.to_csv(args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--products')
    parser.add_argument('--matches')
    parser.add_argument('--output')
    args = parser.parse_args()

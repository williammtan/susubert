from kfp.components import OutputPath, InputPath
import os

def preprocess(input_path: InputPath(str), products_path: OutputPath(str), master_products_path: OutputPath(str)):
    import pandas as pd
    from pathlib import Path

    products = pd.read_csv(input_path).drop_duplicates(subset='id')
    products = products.dropna(subset=['id', 'name', 'description'])
    master_products = products.dropna(subset=['master_product'])

    Path(products_path).parent.mkdir(parents=True, exist_ok=True)
    Path(master_products_path).parent.mkdir(parents=True, exist_ok=True)
    products.to_csv(products_path, index=False)
    master_products.to_csv(master_products_path, index=False)

def train_test_split(
    matches_path: InputPath('str'),
    train_path: OutputPath(str),
    test_path: OutputPath(str),
    test_split: float=0.2,
    ):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from pathlib import Path
    
    matches = pd.read_csv(matches_path)
    train, test = train_test_split(matches, test_size=test_split)
    
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)


def fin(
    matches_path: InputPath(str), 
    products_path: InputPath(str), 
    fin_path: OutputPath(str)
    ):
    from igraph import Graph
    from pathlib import Path

    import pandas as pd
    import numpy as np

    matches = pd.read_csv(matches_path)
    products = pd.read_csv(products_path)

    g = Graph()
    g.add_vertices(len(products))

    id_mapping = dict(zip(products.id, range(len(products))))
    match_pairs = np.array([(id_mapping[match.id1], id_mapping[match.id2]) for _, match in matches.iterrows() if match.match == 1])
    g.add_edges(match_pairs)

    g.vs['id'] = products.id.values
    g.vs['name'] = products.id.name

    clusters = []
    for i, c in enumerate(g.clusters()):
        if len(c) > 2:
            for p in c:
                clusters.append({
                    "id": g.vs[p].attributes()['id'],
                    "cluster": i
            })
    clusters = pd.DataFrame(clusters)

    Path(fin_path).parent.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(fin_path, index=False)


def query_rds(
    save_query_path: OutputPath(str),
    query: str='SELECT * FROM master_product_clusters'
    ):
    from sqlalchemy import create_engine
    from pathlib import Path
    import pandas as pd

    db_connection_str = os.environ['sql_endpoint']
    db_connection = create_engine(db_connection_str)
    db_connection.connect()
    df = pd.read_sql(query, con=db_connection)

    Path(save_query_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_query_path, index=False)

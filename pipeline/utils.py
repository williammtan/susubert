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
    fin_path: OutputPath(str),
    min_cluster_size: int=3
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
    cluster_i = 0
    for c in g.clusters():
        if len(c) > min_cluster_size:
            for p in c:
                clusters.append({
                    "id": g.vs[p].attributes()['id'],
                    "cluster": cluster_i
            })
            cluster_i += 1
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

def save_to_rds(
    dataframe_path: InputPath(str),
    table: str,
    if_exists: str='append',
    index: bool=False
    ):
    from sqlalchemy import create_engine
    import pandas as pd

    db_connection_str = os.environ['sql_endpoint']
    db_connection = create_engine(db_connection_str)
    db_connection.connect()

    df = pd.read_csv(dataframe_path)
    df.to_sql(table, db_connection, schema='food', if_exists=if_exists, index=index)

def drop_cache_matches(
    blocked_matches_path: InputPath(str),
    cached_matches_path: InputPath(str),
    match_candidates_path: OutputPath(str)
    ):
    import pandas as pd

    matches = pd.read_csv(blocked_matches_path)
    cached_matches = pd.read_csv(cached_matches_path)

    match_candidates = matches[~matches.isin(cached_matches)] # find matches that have NOT been cached

    match_candidates.to_csv(match_candidates_path, index=False)

def merge_cache_matches(
    matches_path: InputPath(str),
    cached_matches_path: InputPath(str),
    matches_output_path: OutputPath(str)
    ):
    import pandas as pd
    from pathlib import Path

    df = pd.concat([matches_path, cached_matches_path])

    Path(matches_output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(matches_output_path, index=False)

def save_clusters(
    clusters_path: InputPath(str),
    products_path: InputPath(str)
    ):
    from sqlalchemy import create_engine, update
    import pandas as pd

    db_connection_str = os.environ['sql_endpoint']
    db_connection = create_engine(db_connection_str)
    db_connection.connect()

    query_rds = lambda query: pd.read_sql(query, con=db_connection)

    # load master products and clusters
    df_mp = query_rds("SELECT id, name FROM food.master_products WHERE is_deleted=0").rename(columns={'id':'master_product_id', 'name':'master_product'})
    df_mpc = query_rds("SELECT * FROM food.master_product_clusters WHERE master_product_status_id != 4") # select all mpc that are in use

    # merge products with clusters
    products = pd.read_csv(products_path)
    df_cls = pd.read_csv(clusters_path).merge(products[['id', 'name', 'master_product']], how='left', on='id')
    df_cls = df_cls.merge(df_mp, how="left", on="master_product") # merge with mps
    df_cls.drop_duplicates(subset="id", inplace=True)

    # add statuses
    df_cls["master_product_status_id"] = 1 # unnamed cluster
    df_cls.loc[df_cls.master_product_id.notnull(), "master_product_status_id"] = 2 # named cluster

    # rename to db schema
    df_cls = df_cls.rename(columns={"id":"product_source_id", "cluster":"cluster_id"})
    df_cls = df_cls[["cluster_id", "product_source_id", "master_product_id", "master_product_status_id"]]

    # change status of all the mpcs
    stmt = (
        update('master_product_clusters')
        .values(master_product_status_id=4) # set all mpcs to NOT USED
    )
    with db_connection.begin() as conn:
        conn.execute(stmt) # run update query

    # append to db
    df_cls.to_sql("master_product_clusters", db_connection, schema="food", if_exists="append", index=False)

    # add to mp history
    df_mph_current = df_mpc.copy()
    df_mph_current["current_status_id"] = 4 # changed old clusters status id from UNNAMED OR NAMED CLUSTER to NOT USED

    df_mph_new = df_cls.copy()
    df_mph_new = df_mph_new["master_product_status_id"] # don't change status, meaning added new clusters

    df_mph = pd.concat([df_mph_new, df_mph_new])
    df_mph = df_mph.rename(columns={"id":"master_product_cluster_id", "master_product_status_id":"previous_status_id"})
    df_mph.to_sql("master_product_status_histories", db_connection, schema="food", if_exists="append", index=False)


def download_model(
    model_id: int,
    model_save: OutputPath(str)
    ):
    from sqlalchemy import create_engine
    from google.cloud import storage
    import pandas as pd
    import re

    db_connection_str = os.environ['sql_endpoint']
    db_connection = create_engine(db_connection_str)
    db_connection.connect()

    model_blob = pd.read_sql(f"SELECT * FROM ml_models WHERE id = {model_id}").blob
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(re.search(r'gs://(.*)/', model_blob).group(1))
    blob = bucket.blob(model_blob)
    blob.download_to_filename(model_save)

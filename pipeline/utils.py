from kfp.components import OutputPath, InputPath, create_component_from_func, func_to_container_op
from kubernetes.client.models import V1EnvVar
from typing import Callable, Optional, List
import functools

from dotenv import load_dotenv
import os

load_dotenv()

def _component(func: Optional[Callable] = None,
              *,
              base_image: Optional[str] = None,
              packages_to_install: List[str] = None,
              output_component_file: Optional[str] = None):
    
    if func is None:
        return functools.partial(_component,
                                base_image=base_image,
                                packages_to_install=packages_to_install,
                                output_component_file=output_component_file)
    
   

    comp = func_to_container_op(
      func,
      base_image=base_image,
      packages_to_install=packages_to_install,
      output_component_file=output_component_file)

    if 'sqlalchemy' in packages_to_install:
        def env_wrapper(*args, **kwargs):
            return comp(*args, **kwargs).add_env_variable(V1EnvVar(name="SQL_ENDPOINT", value=os.environ['SQL_ENDPOINT']))

        return env_wrapper

    return comp

def connect_db():
    from sqlalchemy import create_engine
    from pathlib import Path
    import pandas as pd
    import os

    db_connection_str = os.environ['SQL_ENDPOINT']
    db_connection = create_engine(db_connection_str)
    db_connection.connect()
    return db_connection


@_component(
    packages_to_install=['pandas']
)
def preprocess(input_path: InputPath(str), products_path: OutputPath(str), master_products_path: OutputPath(str)):
    import pandas as pd
    from pathlib import Path

    products = pd.read_csv(input_path).drop_duplicates(subset='id')
    products = products.dropna(subset=['id', 'name', 'description'])

    if 'master_product' in products.columns:
        Path(master_products_path).parent.mkdir(parents=True, exist_ok=True)
        master_products = products.dropna(subset=['master_product'])
        master_products.to_csv(master_products_path, index=False)
    else:
        open(master_products_path, 'w')

    Path(products_path).parent.mkdir(parents=True, exist_ok=True)
    products.to_csv(products_path, index=False)

@_component(
    packages_to_install=['pandas']
)
def product_regex(products_path: InputPath(str), output_path: OutputPath(str)):
    import pandas as pd
    from pathlib import Path
    import re

    def get_digits(match):
        digits = re.search('\d+', match.group())
        if digits:
            return int(digits.group())
        else:
            return None

    def re_quantity(string):
        quantity = re.search(r'(isi(?: |)\d+|x(?: |)\d+|\d+(?: |)(?:pcs|pc|kaleng|sachet|pack|(?: |)x)) ', string)
        if not quantity:
            quantity = re.search(r'\d+(?: |)(?:karton|dus)', string)
            if quantity:
                quantity = 12
            else:
                quantity = 1
        else:
            quantity = get_digits(quantity)
        return quantity

    def re_volume(string):
        volume = re.search(r'\d{2,}(?: |)ml ', string)
        if not volume:
            volume = re.search(r'\d+(?: |)(?:l|lt|ltr|liter) ', string)
            if volume:
                volume = get_digits(volume) * 1000
            else:
                volume = None
        else:
            volume = get_digits(volume)
            return volume

    def re_weight(string):
        weight = re.search(r'\d{2,}(?: |)(?:g|gr|gram) ', string)
        if not weight:
            weight = re.search(r'\d+(?: |)(?:|kg|kilogram|kilo) ', string)
            if weight:
                weight = get_digits(weight) * 1000
            else:
                weight = None
        else:
            weight = get_digits(weight)
        return weight


    products = pd.read_csv(products_path)
    products['name_description'] = (products['name'] + ' ' + products['description']).str.lower()

    products['quantity'] = products.name_description.apply(re_quantity)
    products['volume'] = products.name_description.apply(re_volume).astype('Int64')
    products['weight'] = products.name_description.apply(re_weight).astype('Int64')
    products.drop(columns='name_description')
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    products.to_csv(output_path, index=False)


@_component(
    packages_to_install=['pandas', 'sklearn']
)
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

@_component(
    packages_to_install=['python-igraph', 'numpy', 'pandas']
)
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

@_component(
    packages_to_install=['sqlalchemy', 'pandas', 'pymysql']
)
def query_rds(
    save_query_path: OutputPath(str),
    query: str='SELECT * FROM master_product_clusters'
    ):
    from sqlalchemy import create_engine
    from pathlib import Path
    import pandas as pd
    import os

    db_connection_str = os.environ['SQL_ENDPOINT']
    db_connection = create_engine(db_connection_str)
    db_connection.connect()
    df = pd.read_sql(query, con=db_connection)

    Path(save_query_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_query_path, index=False)

@_component(
    packages_to_install=['sqlalchemy', 'pandas', 'pymysql']
)
def save_to_rds(
    dataframe_path: InputPath(str),
    table: str,
    if_exists: str='append',
    index: bool=False
    ):
    from sqlalchemy import create_engine
    import pandas as pd
    import os

    db_connection_str = os.environ['SQL_ENDPOINT']
    db_connection = create_engine(db_connection_str)
    db_connection.connect()

    df = pd.read_csv(dataframe_path)
    df.to_sql(table, db_connection, schema='food', if_exists=if_exists, index=index)

@_component(
    packages_to_install=['pandas']
)
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

@_component(
    packages_to_install=['pandas']
)
def merge_cache_matches(
    matches_path: InputPath(str),
    cached_matches_path: InputPath(str),
    matches_output_path: OutputPath(str)
    ):
    import pandas as pd
    from pathlib import Path

    matches = pd.read_csv(matches_path)
    cached_mataches = pd.read_csv(cached_matches_path)
    df = pd.concat([matches, cached_mataches])

    Path(matches_output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(matches_output_path, index=False)

@_component(
    packages_to_install=['sqlalchemy', 'pandas', 'pymysql']
)
def save_clusters(
    clusters_path: InputPath(str),
    products_path: InputPath(str)
    ):
    from sqlalchemy import create_engine, update
    import pandas as pd
    import os

    db_connection_str = os.environ['SQL_ENDPOINT']
    db_connection = create_engine(db_connection_str)
    db_connection.connect()

    query_rds = lambda query: pd.read_sql(query, con=db_connection)

    # load master products and clusters
    df_mpc = query_rds("SELECT * FROM food.master_product_clusters WHERE master_product_status_id != 4") # select all mpc that are in use

    # merge products with clusters
    products = pd.read_csv(products_path)
    df_cls = pd.read_csv(clusters_path).merge(products[['id', 'name', 'master_product']], how='left', on='id')
    df_cls.drop_duplicates(subset="id", inplace=True)

    # add statuses
    df_cls["master_product_status_id"] = 1 # unnamed cluster
    df_cls.loc[df_cls.master_product.notnull(), "master_product_status_id"] = 2 # named cluster

    # rename to db schema
    df_cls = df_cls.rename(columns={"id":"product_source_id", "cluster":"cluster_id", "master_product": "master_product_id"})
    df_cls = df_cls[["cluster_id", "product_source_id", "master_product_id", "master_product_status_id"]]

    # change status of all the mpcs
    # stmt = (
    #     update('master_product_clusters')
    #     .values(master_product_status_id=4) # set all mpcs to NOT USED
    # )
    # with db_connection.begin() as conn:
    #     conn.execute(stmt) # run update query

    # append to db
    # df_cls.to_sql("master_product_clusters", db_connection, schema="food", if_exists="append", index=False)
    print(df_cls)

    # add to mp history
    df_mph_current = df_mpc[['id', 'master_product_status_id']]
    df_mph_current["current_status_id"] = 4 # changed old clusters status id from UNNAMED OR NAMED CLUSTER to NOT USED

    df_mph_new = pd.read_sql(f"SELECT id, master_product_status_id ORDER BY created_at DESC LIMIT {len(df_cls)}", con=db_connection)
    df_mph_new['current_status_id'] = df_mph_new["master_product_status_id"] # don't change status, meaning added new clusters

    df_mph = pd.concat([df_mph_new, df_mph_new])
    df_mph = df_mph.rename(columns={"id":"master_product_cluster_id", "master_product_status_id":"previous_status_id"})
    # df_mph.to_sql("master_product_status_histories", db_connection, schema="food", if_exists="append", index=False)
    print(df_mph)

@_component(
    packages_to_install=['sqlalchemy', 'pandas', 'pymysql', 'google-cloud-storage']
)
def download_model(
    model_id: int,
    model_save: OutputPath(str)
    ):
    from sqlalchemy import create_engine
    from google.cloud import storage
    from pathlib import Path
    import pandas as pd
    import os

    db_connection_str = os.environ['SQL_ENDPOINT']
    db_connection = create_engine(db_connection_str)
    db_connection.connect()

    model_blob = pd.read_sql(f"SELECT * FROM food.ml_models WHERE id = {model_id}", db_connection).iloc[0].blob
    bucket_name = model_blob.split('/')[2]
    blob_name = model_blob.split(bucket_name + '/')[-1]

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    if len(list(bucket.list_blobs(prefix=blob_name))) == 0:
        # checks if the blob exists (folder or file)
        raise Exception('Model blob doesn\'t exists')
    
    for b in bucket.list_blobs(prefix=blob_name):
        local_path = model_save + b.name.replace(blob_name, '')
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        b.download_to_filename(local_path)

@_component(
    packages_to_install=['sqlalchemy', 'pandas', 'pymysql', 'google-cloud-storage']
)
def save_cache_matches(cache_matches: InputPath(str), model_id):
    from sqlalchemy import create_engine
    import pandas as pd
    import os

    db_connection_str = os.environ['SQL_ENDPOINT']
    db_connection = create_engine(db_connection_str)
    db_connection.connect()

    matches = pd.read_csv(cache_matches).rename({'id1': 'product_source_id_1', 'id2': 'product_source_id_2'})
    matches['model_id'] = model_id

    matches.to_sql('matches_cache', db_connection, schema="food", if_exists='append', index=False)

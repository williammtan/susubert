import pandas as pd
from sqlalchemy import create_engine, update
from google.cloud import storage, secretmanager
import tempfile
from kfp import Client
import os

storage_client = storage.Client()
tmp_dir = tempfile.mkdtemp()

def add_model_db(con_db, blob, tag):

    model_df = pd.DataFrame({
        "name": blob.split('/')[-1],
        "blob": blob,
        "in_use": True,
        "tag": tag
    })
    model_df.to_sql('ml_models', con=con_db, schema='food', if_exists='append', index=False)

    # make other models with susubert tag Not in_use
    stmt = (
        update('ml_models')
        .where(tag='susubert')
        .values(in_use=False)
    )
    with con_db.begin() as conn:
                conn.execute(stmt) # run update query
    
    # return id, since it is auto increment, the id is the len + 1
    return get_len(con_db, 'SELECT * FROM ml_models') + 1

def get_len(con_db, query):
    count_df = pd.read_sql(query, con_db)
    return count_df.iloc[0]['count']

def get_secret(secret='rds-sql-endpoint'):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/501973744948/secrets/{secret}/versions/latest"
    return client.access_secret_version(request={"name": name}).payload.data.decode("UTF-8")

def create_db_connection():
    con_db = create_engine(get_secret())
    con_db.connect()
    return con_db

def run_pipeline(pipeline_package, arguments={}):
    pipeline_filename = os.path.join(tmp_dir, pipeline_package)
    bucket = storage_client.bucket(os.environ.get('BUCKET'))
    pipeline_blob = bucket.blob(os.path.join(os.environ.get('PIPELINES_BLOB'), pipeline_package))
    pipeline_blob.download_to_filename(pipeline_filename)

    kfp_client = Client(host=os.environ.get('KFP_HOST'))
    kfp_client.create_run_from_pipeline_package(pipeline_filename, arguments=arguments)


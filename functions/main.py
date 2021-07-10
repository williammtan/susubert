import random
import os

from utils import *
from constants import *

# ex: request = {"min_update_change": 100}

def update_clusters(request, context):
    """Checks for additional products or retraining, if so, runs match pipeline"""

    con_db = create_db_connection()
    actual_len = get_len(train_query, con=con_db)

    bucket = storage_client.bucket(os.environ.get('BUCKET'))
    min_len_blob_path = os.path.join(os.environ.get('LENGTHS_BLOB'), 'external_temp_products_pareto.txt')
    min_len_blob = bucket.blob(min_len_blob_path)

    matcher_model_id = pd.read_sql('SELECT * FROM food.ml_models WHERE tag = "susubert" ORDER BY created_at DESC LIMIT 1').iloc[0].id
    blocker_model_id = pd.read_sql('SELECT * FROM food.ml_models WHERE tag = "sbert" ORDER BY created_at DESC LIMIT 1').iloc[0].id

    if min_len_blob.exists() and min_len_blob.download_as_string() is not None:
        # download file
        latest_len = int(min_len_blob.download_as_string())

        if actual_len - latest_len > request.get('min_update_change'):
            # if db has been added or changed more than MIN_UPDATE_CHANGE, run the match pipeline
            run_pipeline('match_pipeline.yaml', arguments={'model_id': matcher_model_id, 'sbert_model_id': blocker_model_id})

            return "succeeded in starting run"

    else:
        min_len_blob.upload_from_string(str(actual_len))

        return "products or models not updated"

def update_training(request, context):
    """If additional master products train model"""

    con_db = create_db_connection()

    actual_len = get_len(match_query, con_db)
    bucket_name = os.environ.get('BUCKET')
    bucket = storage_client.bucket(bucket_name)

    min_len_blob_path = os.path.join(os.environ.get('LENGTHS_BLOB'), 'master_products.txt')
    min_len_blob = bucket.blob(min_len_blob_path)

    if min_len_blob.exists() and min_len_blob.download_as_string() is not None:
        # download file
        latest_len = int(min_len_blob.download_as_string())

        if actual_len - latest_len > request.get('min_update_change'):
            # if db has been added or changed more than the MIN_UPDATE_CHANGE, run the match pipeline
            model_blob = f'gs://{bucket_name}/models/susubert/model{random.randint(100000, 999999)}'
            run_pipeline('train_pipeline.yaml', arguments={
                "product_query": "match_query", "model_save": model_blob
            })


            return add_model_db(con_db, model_blob, "susubert") # add model to db and return the id
    else:
        min_len_blob.upload_from_string(str(actual_len))

        latest_model = pd.read_sql("""
                SELECT *
                FROM food.ml_models
                WHERE tag = "susubert"
                ORDER BY created_at DESC
                LIMIT 1;
        """, con_db).iloc[0]
        
        return latest_model.id

def ping_rds_test(request):
    try:
        con_db = create_db_connection()
        return 'connected successfully'
    except Exception as e:
        print(e)
        return f'error: {e}'

import random
from datetime import datetime
from google.cloud import storage, secretmanager
from sqlalchemy import create_engine
from kfp import Client
import pandas as pd
import tempfile
import base64
import json
import os

from constants import *

kfp_client = Client(host=os.environ.get('KFP_HOST'))
storage_client = storage.Client()
tmp_dir = tempfile.mkdtemp()

client = secretmanager.SecretManagerServiceClient()
db_connection_str = client.access_secret_version(request={"name": f"projects/501973744948/secrets/{os.environ['MYSQL_SECRET_NAME']}/versions/latest"}).payload.data.decode("UTF-8")
db_connection = create_engine(db_connection_str)

def update(event, context):
    data = json.loads(base64.b64decode(event['data']).decode('utf-8'))
    task = Task(pipeline=data.get('pipeline'), query=data.get('query'), min_difference=data.get('min_difference'), params=data.get('params'))
    return task()

class Task:
    def __init__(self, pipeline, query, min_difference, params):
        """
        Params:
            - pipeline: name of pipeline inside the pipeline directory gs://foodid_product_matching/pipelines/
            - query: count query to check the difference
            - min_difference: minimum number of new objects to start the run
            - params: dict containing pipeline parameters
        """
        self.pipeline = pipeline
        self.query = query
        self.min_difference = min_difference
        self.params = params

        self.bucket = storage_client.bucket(os.environ['BUCKET'])
        self.count_blob = self.bucket.blob(os.path.join(os.environ['LENGTHS_BLOB'], self.pipeline))
        self.pipeline_blob = self.bucket.blob(os.path.join(os.environ['PIPELINES_BLOB'], self.pipeline))
    
    def count(self):
        """Get the count of the query"""
        df = pd.read_sql(self.query, db_connection)
        return int(df.iloc[0, 0])
    
    def get_count_blob(self):
        """Get the value of the count blob, returns None if its empty, not a real value or non-existant"""
        if self.count_blob.exists():
            value = self.count_blob.download_as_string()
            try:
                value = int(value)
            except ValueError:
                value = None
        else:
            value = None
        return value
    
    def update_count_blob(self, count):
        self.count_blob.upload_from_string(str(count))
    
    def run_pipeline(self):
        pipeline_filename = os.path.join(tmp_dir, self.pipeline)
        self.pipeline_blob.download_to_filename(pipeline_filename)

        return kfp_client.create_run_from_pipeline_package(pipeline_filename, arguments=self.params)

    
    def __call__(self):
        count = self.count()
        last_count = self.get_count_blob()

        if last_count is None or (count - last_count >= self.min_difference):
            # run the pipeline
            result = self.run_pipeline()
            self.update_count_blob(count)
            return {'run': True, 'run_id': result.run_id}
        else:
            return {'run': False}

        

# task = Task(pipeline="oneshot_match_pipeline.yaml", query=train_query, min_difference=100, params={
# {
#     "lm": "indobenchmark/indobert-base-p1",
#     "model_id": 15,
#     "sbert_model_id": 14,
#     "batch_size": 32,
#     "product_query": "SELECT p.id, pf.master_product_id, p.name, p.outlet_id, p.description, p.weight, FORMAT(FLOOR(pp.price),0) AS price, pc_main.id AS product_category_id, pc_main.name AS main_category, pc_sub.name AS sub_category FROM products p LEFT JOIN product_fins pf ON pf.product_id = p.id JOIN product_prices pp ON p.id = pp.product_id JOIN product_category_children pcc ON p.product_category_id = pcc.child_category_id JOIN product_categories pc_main ON pc_main.id = pcc.product_category_id JOIN product_categories pc_sub ON pc_sub.id = pcc.child_category_id WHERE pc_main.name IN (SELECT pc_main.name AS main_category FROM products p JOIN product_fins pf ON pf.product_id = p.id JOIN product_prices pp ON p.id = pp.product_id JOIN product_category_children pcc ON p.product_category_id = pcc.child_category_id JOIN product_categories pc_main ON pc_main.id = pcc.product_category_id JOIN product_categories pc_sub ON pc_sub.id = pcc.child_category_id GROUP BY pc_main.name HAVING count(pc_main.name) > 4) AND (pf.master_product_id IS NULL OR pf.is_removed = 1)",
#     "blocker_top_k": 10,
#     "blocker_threshold": 0.4,
#     "match_threshold": 0.8
# }


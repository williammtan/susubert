import kfp 
from kfp import dsl
from kfp.components import create_component_from_func
from dotenv import load_dotenv
import os

from utils import query_rds

load_dotenv()

def gpu_check_op():
    return dsl.ContainerOp(
        name='check_gpu',
        image='tensorflow/tensorflow:latest-gpu',
        command=['sh', '-c'],
        arguments=['nvidia-smi']
    ).set_gpu_limit(1)

def ip_check_op():
    return dsl.ContainerOp(
        name='check_ips',
        image='python:3',
        command=['curl', 'http://httpbin.org/ip']
    )

def connect_to_rds_op():
    import pandas as pd
    from sqlalchemy import create_engine
    db_connection_str = os.environ['sql_endpoint']
    db_connection = create_engine(db_connection_str)
    db_connection.connect()
    df = pd.read_sql('SELECT * FROM master_product_clusters', con=db_connection)
    print(df)

@dsl.pipeline(
    name='GPU check',
)
def gpu_pipeline():
    gpu_check = gpu_check_op()
    ip_check = ip_check_op()
    query_rds_task = query_rds()

if __name__ == '__main__':
    client = kfp.Client(host='https://2db1ee630a900d84-dot-us-central1.pipelines.googleusercontent.com')
    client.create_run_from_pipeline_func(gpu_pipeline, arguments={})

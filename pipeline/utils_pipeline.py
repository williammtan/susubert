import kfp 
from kfp import dsl
from kfp.components import create_component_from_func

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
    db_connection_str = '***REMOVED***'
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
    connect_to_rds = create_component_from_func(func=connect_to_rds_op, packages_to_install=['pandas', 'SQLAlchemy', 'pymysql'])()

if __name__ == '__main__':
    client = kfp.Client(host='https://118861cf2b92c13d-dot-us-central1.pipelines.googleusercontent.com')
    client.create_run_from_pipeline_func(gpu_pipeline, arguments={})

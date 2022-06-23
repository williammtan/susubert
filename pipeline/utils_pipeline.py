import kfp 
from kfp import dsl
from kfp.components import create_component_from_func
from kubernetes.client.models import V1EnvVar
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


@dsl.pipeline(
    name='GPU check',
)
def gpu_pipeline():
    ip_check = ip_check_op().set_gpu_limit(1)
    ip_check_op().after(ip_check)
    query_rds().after(ip_check)
    gpu_check_op()

if __name__ == '__main__':
    client = kfp.Client(host='https://2286482f38de0564-dot-us-central1.pipelines.googleusercontent.com')
    client.create_run_from_pipeline_func(gpu_pipeline, arguments={})

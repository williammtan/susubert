import kfp 
from kfp import dsl

def gpu_check_op():
    return dsl.ContainerOp(
        name='check',
        image='tensorflow/tensorflow:latest-gpu',
        command=['sh', '-c'],
        arguments=['nvidia-smi']
    ).set_gpu_limit(1)

@dsl.pipeline(
    name='GPU check',
)
def gpu_pipeline():
    gpu_check = gpu_check_op()

if __name__ == '__main__':
    client = kfp.Client(host='https://118861cf2b92c13d-dot-us-central1.pipelines.googleusercontent.com')
    client.create_run_from_pipeline_func(gpu_pipeline, arguments={})

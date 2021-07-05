import kfp
from kfp import dsl
from kfp.components import func_to_container_op, load_component_from_url, load_component_from_file
import sys

from utils import preprocess

@dsl.pipeline(name='train pipeline')
def match_pipeline(
    lm:str='indobenchmark/indobert-base-p1',
    products:'URI'='gs://ml_foodid_project/product-matching/susubert/pareto_training.csv', # type: ignore
    model_save: str='gs://ml_foodid_project/product-matching/susubert/pareto_model',
    keep_columns: list=['name', 'price'],

    batch_size: int=32,

    blocker_epochs: int=2,
    blocker_learning_rate: float=2e-5,
    blocker_threshold: float=0.5,
    blocker_top_k: int=100,

    match_threshold: float=0.8
):
    download_op = load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/0795597562e076437a21745e524b5c960b1edb68/components/google-cloud/storage/download/component.yaml')
    preprocess_op = func_to_container_op(func=preprocess, packages_to_install=['pandas'])
    feature_extraction_op = load_component_from_file('feature_extraction/component.yaml') 
    batch_selection_op = load_component_from_file('batch_selection/component.yaml')
    serialize_op = load_component_from_file('serialize/component.yaml')
    train_blocker_op = load_component_from_file('train_blocker/component.yaml')
    upload_op = load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/master/components/google-cloud/storage/upload_to_explicit_uri/component.yaml')

    # download and simple preprocess
    download_task = download_op(products)
    preprocess_task = preprocess_op(download_task.output)

    # preprocessing
    feature_extraction_task = feature_extraction_op(lm, preprocess_task.outputs['master_products']).set_gpu_limit(1)
    batch_selection_task = batch_selection_op(preprocess_task.outputs['master_products'], feature_extraction_task.output).set_gpu_limit(1)
    serialize_task = serialize_op(batch_selection_task.output, preprocess_task.outputs['master_products'], keep_columns).set_gpu_limit(1)
    train_blocker_task = train_blocker_op(serialize_task.output, lm, batch_size, blocker_learning_rate, blocker_epochs).set_gpu_limit(1)



if __name__ == '__main__':
    if sys.argv[1] == 'compile':
        kfp.compiler.Compiler().compile(match_pipeline, 'match_pipeline.yaml')
    elif sys.argv[1] == 'run':
        client = kfp.Client(host='https://118861cf2b92c13d-dot-us-central1.pipelines.googleusercontent.com')
        client.create_run_from_pipeline_func(match_pipeline, arguments={})

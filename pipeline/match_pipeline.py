import kfp
from kfp import dsl
from kfp.components import func_to_container_op, load_component_from_url, load_component_from_file
import sys

from utils import preprocess, fin

@dsl.pipeline(name='train pipeline')
def match_pipeline(
    lm:str='indobenchmark/indobert-base-p1',
    products:'URI'='gs://ml_foodid_project/product-matching/susubert/pareto_training.csv', # type: ignore
    model_save: 'URI'='gs://ml_foodid_project/product-matching/susubert/pareto_model/model', # type: ignore
    keep_columns: list=['name', 'price'],

    batch_size: int=32,

    blocker_epochs: int=1,
    blocker_learning_rate: float=2e-5,
    blocker_threshold: float=0.5,
    blocker_top_k: int=2,

    match_threshold: float=0.8
):
    download_op = load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/0795597562e076437a21745e524b5c960b1edb68/components/google-cloud/storage/download/component.yaml')
    preprocess_op = func_to_container_op(func=preprocess, packages_to_install=['pandas'])
    feature_extraction_op = load_component_from_file('feature_extraction/component.yaml') 
    batch_selection_op = load_component_from_file('batch_selection/component.yaml')
    serialize_op = load_component_from_file('serialize/component.yaml')
    train_blocker_op = load_component_from_file('train_blocker/component.yaml')
    blocker_op = load_component_from_file('blocker/component.yaml')
    matcher_op = load_component_from_file('matcher/component.yaml')
    fin_op = func_to_container_op(func=fin, packages_to_install=['python-igraph', 'numpy', 'pandas'])
    upload_op = load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/master/components/google-cloud/storage/upload_to_explicit_uri/component.yaml')

    # download and simple preprocess
    download_task = download_op(products)
    preprocess_task = preprocess_op(download_task.output)

    # preprocessing
    feature_extraction_task = feature_extraction_op(lm, preprocess_task.outputs['master_products']).set_gpu_limit(1)
    batch_selection_task = batch_selection_op(preprocess_task.outputs['master_products'], feature_extraction_task.output).set_gpu_limit(1)
    serialize_task = serialize_op(matches=batch_selection_task.output, products=preprocess_task.outputs['master_products'], keepcolumns=keep_columns).set_gpu_limit(1)
    serialize_products_task = serialize_op(matches='', products=preprocess_task.outputs['products'], keepcolumns=keep_columns).set_gpu_limit(1)

    # blocking
    train_blocker_task = train_blocker_op(serialize_task.output, lm, batch_size, blocker_learning_rate, blocker_epochs).set_gpu_limit(1)
    blocker_task = blocker_op(preprocess_task.outputs['products'], serialize_products_task.output, train_blocker_task.output, blocker_top_k, blocker_threshold).set_gpu_limit(1)
    serialize_blocker_task = serialize_op(matches=blocker_task.output, products=preprocess_task.outputs['products'], keepcolumns=keep_columns).set_gpu_limit(1)

    # matching
    download_model_task = download_op(model_save)
    matcher_task = matcher_op(matches=serialize_blocker_task.output, lm=lm, model=download_model_task.output, batchsize=batch_size, threshold=match_threshold).set_gpu_limit(1)
    fin_task = fin_op(matches=matcher_task.output, products=preprocess_task.outputs['products'])



if __name__ == '__main__':
    if sys.argv[1] == 'compile':
        kfp.compiler.Compiler().compile(match_pipeline, 'match_pipeline.yaml')
    elif sys.argv[1] == 'run':
        client = kfp.Client(host='https://118861cf2b92c13d-dot-us-central1.pipelines.googleusercontent.com')
        client.create_run_from_pipeline_func(match_pipeline, arguments={})

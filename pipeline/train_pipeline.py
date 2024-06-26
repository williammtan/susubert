import kfp
from kfp import dsl
from kfp.components import load_component_from_url, load_component_from_file
import sys

from utils import preprocess, train_test_split, query_rds

@dsl.pipeline(name='train pipeline')
def train_pipeline(
    lm:str='indobenchmark/indobert-base-p1',
    product_query:str="""SELECT * from food.master_product_clusters WHERE master_product_status_id = 2""",
    model_save: str='gs://ml_foodid_project/product-matching/susubert/pareto_model',
    keep_columns: list=['name', 'price'],
    batch_size: int=32,
    learning_rate: float=2e-5,
    num_epochs: int=2
):
    feature_extraction_op = load_component_from_file('feature_extraction/component.yaml') 
    batch_selection_op = load_component_from_file('batch_selection/component.yaml')
    serialize_op = load_component_from_file('serialize/component.yaml')
    train_op = load_component_from_file('train/component.yaml')
    evaluate_op = load_component_from_file('evaluate/component.yaml')
    upload_op = load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/c783705c0e566c611ef70160a01e3ed0865051bd/components/contrib/google-cloud/storage/upload_to_explicit_uri/component.yaml')

    # download and simple preprocess
    query_op = query_rds(query=product_query)
    preprocess_task = preprocess(query_op.output)

    # preprocessing
    feature_extraction_task = feature_extraction_op(lm, preprocess_task.outputs['master_products']).set_gpu_limit(1)
    batch_selection_task = batch_selection_op(preprocess_task.outputs['master_products'], feature_extraction_task.output).set_gpu_limit(1)
    serialize_op = serialize_op(batch_selection_task.output, preprocess_task.outputs['master_products'], keep_columns)
    train_test_split_task = train_test_split(serialize_op.output)

    # training
    train_task = train_op(train_test_split_task.outputs['train'], lm, batch_size, learning_rate, num_epochs).set_gpu_limit(1)
    evaluate_task = evaluate_op(train_test_split_task.outputs['test'], lm, train_task.output, batch_size).set_gpu_limit(1)
    upload_task = upload_op(train_task.output, model_save)


if __name__ == '__main__':
    if sys.argv[1] == 'compile':
        kfp.compiler.Compiler().compile(train_pipeline, 'train_pipeline.yaml')
    elif sys.argv[1] == 'run':
        client = kfp.Client(host='https://2286482f38de0564-dot-us-central1.pipelines.googleusercontent.com')
        client.create_run_from_pipeline_func(train_pipeline, arguments={'num_epochs':1})
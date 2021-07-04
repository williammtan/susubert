import kfp
from kfp import dsl
from kfp.components import func_to_container_op, load_component_from_url, load_component_from_file

from utils import preprocess, train_test_split

@dsl.pipeline(name='train pipeline')
def train_pipeline(
    lm:str='indobenchmark/indobert-base-p1',
    products:'URI'='gs://ml_foodid_project/product-matching/pareto_training.csv', # type: ignore
    keep_columns: list=['name', 'price']
):
    download_op = load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/0795597562e076437a21745e524b5c960b1edb68/components/google-cloud/storage/download/component.yaml')
    preprocess_op = func_to_container_op(func=preprocess, packages_to_install=['pandas'])
    feature_extraction_op = load_component_from_file('feature_extraction/component.yaml') 
    batch_selection_op = load_component_from_file('batch_selection/component.yaml')
    serialize_op = load_component_from_file('serialize/component.yaml')
    train_test_split_op = func_to_container_op(func=train_test_split, packages_to_install=['pandas', 'sklearn'])

    # download and simple preprocess
    download_task = download_op(products)
    preprocess_task = preprocess_op(download_task.output)

    # preprocessing
    feature_extraction_task = feature_extraction_op(lm, preprocess_task.output).set_gpu_limit(1)
    batch_selection_task = batch_selection_op(preprocess_task.output, feature_extraction_task.output).set_gpu_limit(1)
    serialize_op = serialize_op(batch_selection_task.output, preprocess_task.output, keep_columns).set_gpu_limit(1)
    train_test_split_task = train_test_split_op(serialize_op.output)


if __name__ == '__main__':
    # kfp.compiler.Compiler().compile(train_pipeline, 'train_pipline.yaml')
    client = kfp.Client(host='https://118861cf2b92c13d-dot-us-central1.pipelines.googleusercontent.com')
    client.create_run_from_pipeline_func(train_pipeline, arguments={})

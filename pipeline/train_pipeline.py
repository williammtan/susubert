import kfp
from kfp import dsl
from kfp.components import load_component_from_url, load_component_from_file

@dsl.pipeline(name='train pipeline')
def train_pipeline(
    lm:str='indobenchmark/indobert-base-p1',
    products:'URI'='gs://ml_foodid_project/product-matching/pareto_training.csv'
):
    download_op = load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/0795597562e076437a21745e524b5c960b1edb68/components/google-cloud/storage/download/component.yaml')
    feature_extraction_op = load_component_from_file('feature_extraction/component.yaml')

    download_task = download_op(products)

    feature_extraction_op(lm, download_task.output)

if __name__ == '__main__':
    # kfp.compiler.Compiler().compile(train_pipeline, 'train_pipline.yaml')
    client = kfp.Client(host='https://118861cf2b92c13d-dot-us-central1.pipelines.googleusercontent.com')
    client.create_run_from_pipeline_func(train_pipeline, arguments={})

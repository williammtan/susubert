import kfp
from kfp import dsl
from kfp.components import load_component_from_url, load_component_from_file
import sys

from utils import preprocess, query_rds

@dsl.pipeline(name='train pipeline')
def train_pipeline(
    lm:str='indobenchmark/indobert-base-p1',
    product_query:str="""SELECT p.id, pf.master_product_id, CONVERT(p.name, CHAR) as name, CONVERT(p.description, CHAR) as description, p.weight, FORMAT(FLOOR(pp.price),0) AS price, CONVERT(pc_main.name, CHAR) AS main_category, CONVERT(pc_sub.name, CHAR) AS sub_category FROM products p JOIN product_fins pf ON pf.product_id = p.id JOIN product_prices pp ON p.id = pp.product_id JOIN product_category_children pcc ON p.product_category_id = pcc.child_category_id JOIN product_categories pc_main ON pc_main.id = pcc.product_category_id JOIN product_categories pc_sub ON pc_sub.id = pcc.child_category_id WHERE pf.is_removed = 0 UNION SELECT ext.id_source AS id, mpc.master_product_id, ext.name, ext.description, ext.weight, ext.price, ext.main_category, ext.sub_category FROM master_product_clusters mpc JOIN external_temp_products_pareto ext ON mpc.product_source_id = ext.id_source WHERE mpc.master_product_status_id = 2""",
    model_save: str='gs://ml_foodid_project/product-matching/susubert/sbert_mix_model',
    keep_columns: list=['name', 'price'],
    batch_size: int=32,
    learning_rate: float=2e-5,
    num_epochs: int=2
):
    feature_extraction_op = load_component_from_file('feature_extraction/component.yaml') 
    batch_selection_op = load_component_from_file('batch_selection/component.yaml')
    serialize_op = load_component_from_file('serialize/component.yaml')
    train_op = load_component_from_file('train_blocker/component.yaml')
    upload_op = load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/c783705c0e566c611ef70160a01e3ed0865051bd/components/contrib/google-cloud/storage/upload_to_explicit_uri/component.yaml')

    # download and simple preprocess
    query_op = query_rds(query=product_query)
    preprocess_task = preprocess(query_op.output)

    # preprocessing
    feature_extraction_task = feature_extraction_op(lm, preprocess_task.outputs['master_products']).set_gpu_limit(1)
    batch_selection_task = batch_selection_op(preprocess_task.outputs['master_products'], feature_extraction_task.output).set_gpu_limit(1)
    serialize_op = serialize_op(batch_selection_task.output, preprocess_task.outputs['master_products'], keep_columns)

    # training
    train_task = train_op(serialize_op.output, lm, batch_size, learning_rate, num_epochs).set_gpu_limit(1)
    upload_task = upload_op(train_task.output, model_save)


if __name__ == '__main__':
    if sys.argv[1] == 'compile':
        kfp.compiler.Compiler().compile(train_pipeline, 'blocker_train_pipeline.yaml')
    elif sys.argv[1] == 'run':
        client = kfp.Client(host='https://2286482f38de0564-dot-us-central1.pipelines.googleusercontent.com')
        client.create_run_from_pipeline_func(train_pipeline, arguments={'num_epochs':1})
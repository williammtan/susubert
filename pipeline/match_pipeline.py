import kfp
from kfp import dsl
from kfp.components import load_component_from_file
import sys

from utils import *

@dsl.pipeline(name='matcher pipeline')
def match_pipeline(
    lm: str='indobenchmark/indobert-base-p1',
    model_id: int=1, # type: ignore
    sbert_model_id: int=2, # type: ignore
    batch_size: int=32,

    product_query: str="""
        SELECT prod.id_source as id, prod.name, prod.description, prod.weight, prod.price, prod.main_category, prod.sub_category, mpc.master_product_id as master_product
        FROM food.external_temp_products_pareto prod LEFT JOIN (SELECT * FROM food.master_product_clusters) mpc ON prod.id = mpc.master_product_id
    """,

    keep_columns: list=['name', 'price'],
    blocker_top_k: int=100,
    blocker_threshold: float=0.25,

    match_threshold: float=0.8,
    min_cluster_size: int=3,
):
    """This pipeline will block matches and predict product matches (using cache) to create clusters."""

    serialize_op = load_component_from_file('serialize/component.yaml')
    blocker_op = load_component_from_file('blocker/component.yaml')
    matcher_op = load_component_from_file('matcher/component.yaml')

    download_sbert_task = download_model(sbert_model_id)
    download_model_task = download_model(model_id)

    query_products = query_rds(query=product_query)
    preprocess_task = preprocess(query_products.output)
    re_task = product_regex(preprocess_task.outputs['products'])

    serialize_products_task = serialize_op(matches='', products=re_task.output, keepcolumns=keep_columns)
    blocker_task = blocker_op(re_task.output, serialize_products_task.output, download_sbert_task.output, blocker_top_k, blocker_threshold).set_gpu_limit(1)
    serialize_matches_task = serialize_op(matches=blocker_task.output, products=re_task.output, keepcolumns=keep_columns)

    matcher_task = matcher_op(matches=serialize_matches_task.output, lm=lm, model=download_model_task.output, batchsize=batch_size, threshold=match_threshold).set_gpu_limit(1)    
    fin_task = fin(matches=matcher_task.output, products=re_task.output, min_cluster_size=min_cluster_size).set_gpu_limit(1)
    save_clusters_task = save_clusters(clusters=fin_task.output, products=re_task.output, min_cluster_size=min_cluster_size)

if __name__ == '__main__':
    if sys.argv[1] == 'compile':
        kfp.compiler.Compiler().compile(match_pipeline, 'match_pipeline.yaml')
    elif sys.argv[1] == 'run':
        client = kfp.Client(host='https://2286482f38de0564-dot-us-central1.pipelines.googleusercontent.com')
        client.create_run_from_pipeline_func(match_pipeline, arguments={"blocker_threshold": 0.25, "blocker_top_k": 100})
import kfp
from kfp import dsl
from kfp.components import load_component_from_file
import sys

from utils import *

@dsl.pipeline(name='oneshot matcher pipeline')
def oneshot_match_pipeline(
    lm: str='indobenchmark/indobert-base-p1',
    model_id: int=1, # type: ignore
    sbert_model_id: int=8, # type: ignore
    batch_size: int=32,

    product_query: str="""
        SELECT prod.id_source as id, prod.name, prod.description, prod.weight, prod.price, prod.main_category, prod.sub_category, mpc.master_product_id, mp.name as master_product
        FROM food.external_temp_products_pareto prod LEFT JOIN food.master_product_clusters mpc ON prod.id = mpc.master_product_id LEFT JOIN food.master_products mp ON mp.id = mpc.master_product_id
    """,

    blocker_top_k: int=100,
    blocker_threshold: float=0.25,

    match_threshold: float=0.8,
    cache_matches_table: str="matches_cache",
):
    """This pipeline will block matches and predict product matches (using cache) to create clusters."""

    blocker_op = load_component_from_file('blocker/oneshot_component.yaml')
    matcher_op = load_component_from_file('matcher/component.yaml')

    download_sbert_task = download_model(sbert_model_id)
    download_model_task = download_model(model_id)

    query_products = query_rds(query=product_query)
    preprocess_task = preprocess(query_products.output)
    re_task = product_regex(preprocess_task.outputs['products'])

    blocker_task = blocker_op(re_task.output, download_sbert_task.output, blocker_top_k, blocker_threshold).set_gpu_limit(1)
    matcher_task = matcher_op(matches=blocker_task.output, lm=lm, model=download_model_task.output, batchsize=batch_size, threshold=match_threshold).set_gpu_limit(1)
    generate_suggestions_task = generate_fin_suggestions(matches=matcher_task.output, products=re_task.output).set_gpu_limit(1)

if __name__ == '__main__':
    if sys.argv[1] == 'compile':
        kfp.compiler.Compiler().compile(oneshot_match_pipeline, 'oneshot_match_pipeline.yaml')
    elif sys.argv[1] == 'run':
        client = kfp.Client(host='https://2286482f38de0564-dot-us-central1.pipelines.googleusercontent.com')
        client.create_run_from_pipeline_func(oneshot_match_pipeline, arguments={})

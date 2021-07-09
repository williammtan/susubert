import kfp
from kfp import dsl
from kfp.components import func_to_container_op, load_component_from_url, load_component_from_file

from utils import preprocess, query_rds, save_clusters, fin, drop_cache_matches, merge_cache_matches

@dsl.pipeline(name='matcher pipeline')
def match_pipeline(
    lm: str='indobenchmark/indobert-base-p1',
    model_save: 'URI'='gs://ml_foodid_project/product-matching/susubert/pareto_model/model', # type: ignore
    batch_size: int=32,
    sbert_model: 'URI', # type: ignore

    product_query: str="""
    SELECT *
    FROM external_temp_products_pareto
    WHERE id IN (SELECT id FROM master_product_clusters WHERE master_product_status_id = 3) OR id NOT IN (SELECT id FROM master_product_clusters)
    """,
    
    keep_columns: list=['name', 'price'],
    blocker_top_k: int=100,
    blocker_threshold: float=0.25,

    match_threshold: float=0.8,
    min_cluster_size: int=3,

    cache_matches_table: str="matches_cache",
):
    """This pipeline will block matches and predict product matches (using cache) to create clusters."""

    download_op = load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/0795597562e076437a21745e524b5c960b1edb68/components/google-cloud/storage/download/component.yaml')
    preprocess_op = func_to_container_op(func=preprocess, packages_to_install=['pandas'])
    query_rds_op = func_to_container_op(func=query_rds, packages_to_install=['sqlalchemy', 'pandas', 'pymysql'])
    serialize_op = load_component_from_file('serialize/component.yaml')
    blocker_op = load_component_from_file('blocker/component.yaml')
    drop_cache_op = func_to_container_op(func=drop_cache_matches, packages_to_install=['sqlalchemy', 'pandas', 'pymysql'])
    matcher_op = load_component_from_file('matcher/component.yaml')
    merge_cache_op = func_to_container_op(func=merge_cache_matches, packages_to_install=['pandas'])
    fin_op = func_to_container_op(func=fin, packages_to_install=['python-igraph', 'numpy', 'pandas'])
    save_clusters_op = func_to_container_op(func=save_clusters, packages_to_install=['sqlalchemy', 'pandas', 'pymysql'])

    download_sbert_task = download_op(sbert_model)
    download_model_task = download_op(model_save)

    query_products = query_rds_op(query=product_query)
    preprocess_task = preprocess_op(query_products.output)

    serialize_products_task = serialize_op(matches='', products=preprocess_task.outputs['products'], keepcolumns=keep_columns).set_gpu_limit(1)
    blocker_task = blocker_op(preprocess_task.outputs['products'], serialize_products_task.output, download_sbert_task.output, blocker_top_k, blocker_threshold).set_gpu_limit(1)

    if cache_matches_table is not None:
        query_cache_matches = query_rds_op(query=f"SELECT * FROM {cache_matches_table} WHERE model = {model_save.split('/')[-1]}") # find matches where the model is the inputted model
        drop_cache_task = drop_cache_op(blocked_matches=blocker_task.output, cached_matches=query_cache_matches.output)
        serialize_matches_task = serialize_op(matches=drop_cache_task.output, products=preprocess_task.outputs['products'], keepcolumns=keep_columns).set_gpu_limit(1)
    else:
        serialize_matches_task = serialize_op(matches=blocker_task.output, products=preprocess_task.outputs['products'], keepcolumns=keep_columns).set_gpu_limit(1)

    matcher_task = matcher_op(matches=serialize_matches_task.output, lm=lm, model=download_model_task.output, batchsize=batch_size, threshold=match_threshold).set_gpu_limit(1)

    if cache_matches_table is not None:
        merge_cache_task = merge_cache_op(matches=matcher_task.output, cache_matches=query_cache_matches.output)
        fin_task = fin_op(matches=merge_cache_task.output, products=preprocess_task.outputs['products'], min_cluster_size=min_cluster_size).set_gpu_limit(1)
    else:
        fin_task = fin_op(matches=matcher_task.output, products=preprocess_task.outputs['products'], min_cluster_size=min_cluster_size).set_gpu_limit(1)

    save_clusters_task = save_clusters_op(clusters=fin_task.output)

    
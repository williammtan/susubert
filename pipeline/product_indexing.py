import kfp
from kfp import dsl
from kfp.components import load_component_from_file, load_component_from_url
import sys

from utils import *

@dsl.pipeline(name='product indexing pipeline')
def match_pipeline(
    lm="indobenchmark/indobert-base-p1",
    query="SELECT * FROM master_products"
):
    feature_extraction_op = load_component_from_file('feature_extraction/component.yaml') 

    query_products = query_rds(query=query)
    feature_extraction_task = feature_extraction_op(lm, query_products.output).set_gpu_limit(1)
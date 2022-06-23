import os
import yaml

def setup_environment():
    with open('.env.yaml', 'r') as stream:
        env = yaml.safe_load(stream)
    
    for k, v in env.items():
        os.environ[k] = v

if __name__ == '__main__':
    setup_environment()

    from main import *
    # update_clusters({'min_update_change': 100}, None)
    result = update({'pipeline': 'oneshot_match_pipeline.yaml', 'query': 'SELECT COUNT(*) FROM products p LEFT JOIN product_fins pf ON pf.product_id = p.id JOIN product_prices pp ON p.id = pp.product_id JOIN product_category_children pcc ON p.product_category_id = pcc.child_category_id JOIN product_categories pc_main ON pc_main.id = pcc.product_category_id JOIN product_categories pc_sub ON pc_sub.id = pcc.child_category_id WHERE pc_main.name IN (SELECT pc_main.name AS main_category FROM products p JOIN product_fins pf ON pf.product_id = p.id JOIN product_prices pp ON p.id = pp.product_id JOIN product_category_children pcc ON p.product_category_id = pcc.child_category_id JOIN product_categories pc_main ON pc_main.id = pcc.product_category_id JOIN product_categories pc_sub ON pc_sub.id = pcc.child_category_id GROUP BY pc_main.name HAVING count(pc_main.name) > 4) AND (pf.master_product_id IS NULL OR pf.is_removed = 1)', 'min_difference':100, 'params': { "lm": "indobenchmark/indobert-base-p1", "model_id": 15, "sbert_model_id": 14, "batch_size": 32, "product_query": "SELECT p.id, pf.master_product_id, p.name, p.outlet_id, p.description, p.weight, FORMAT(FLOOR(pp.price),0) AS price, pc_main.id AS product_category_id, pc_main.name AS main_category, pc_sub.name AS sub_category FROM products p LEFT JOIN product_fins pf ON pf.product_id = p.id JOIN product_prices pp ON p.id = pp.product_id JOIN product_category_children pcc ON p.product_category_id = pcc.child_category_id JOIN product_categories pc_main ON pc_main.id = pcc.product_category_id JOIN product_categories pc_sub ON pc_sub.id = pcc.child_category_id WHERE pc_main.name IN (SELECT pc_main.name AS main_category FROM products p JOIN product_fins pf ON pf.product_id = p.id JOIN product_prices pp ON p.id = pp.product_id JOIN product_category_children pcc ON p.product_category_id = pcc.child_category_id JOIN product_categories pc_main ON pc_main.id = pcc.product_category_id JOIN product_categories pc_sub ON pc_sub.id = pcc.child_category_id GROUP BY pc_main.name HAVING count(pc_main.name) > 4) AND (pf.master_product_id IS NULL OR pf.is_removed = 1)", "blocker_top_k": 10, "blocker_threshold": 0.4, "match_threshold": 0.8 }}, None)
    print(result)

    # update_training({'min_update_change': 100}, None)


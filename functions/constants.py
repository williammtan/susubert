
train_query = """
    SELECT product_source_id as id, master_product_id as master_product, name, description, weight, price, main_category, sub_category
    FROM food.master_product_clusters mpc
    LEFT JOIN food.external_temp_products_pareto prod
    ON prod.id_source = mpc.product_source_id
    WHERE master_product_status_id = 2
"""

train_check_query = """
    SELECT COUNT(*) as count
    FROM food.master_product_clusters mpc
    LEFT JOIN food.external_temp_products_pareto prod
    ON prod.id_source = mpc.product_source_id
    WHERE master_product_status_id = 2
"""

match_check_query = """
    SELECT COUNT(*) as count
    FROM food.external_temp_products_pareto
    WHERE id IN (SELECT id FROM food.master_product_clusters WHERE master_product_status_id = 3) OR id NOT IN (SELECT id FROM food.master_product_clusters)
"""

matcher_query = """
    SELECT prod.id_source as id, prod.name, prod.description, prod.weight, prod.price, prod.main_category, prod.sub_category, mpc.master_product_id as master_product
    FROM food.external_temp_products_pareto prod LEFT JOIN (SELECT * FROM food.master_product_clusters) mpc ON prod.id = mpc.master_product_id
"""

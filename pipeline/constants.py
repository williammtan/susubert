train_mixed_sql = """
SELECT p.id,
      pf.master_product_id,
      CONVERT(p.name, CHAR) as name,
      CONVERT(p.description, CHAR) as description,
      p.weight,
      FORMAT(FLOOR(pp.price),0) AS price,
      CONVERT(pc_main.name, CHAR) AS main_category,
      CONVERT(pc_sub.name, CHAR) AS sub_category
FROM products p
JOIN product_fins pf ON pf.product_id = p.id
JOIN product_prices pp ON p.id = pp.product_id
JOIN product_category_children pcc ON p.product_category_id = pcc.child_category_id
JOIN product_categories pc_main ON pc_main.id = pcc.product_category_id
JOIN product_categories pc_sub ON pc_sub.id = pcc.child_category_id
UNION
SELECT ext.id_source AS id,
      mpc.master_product_id,
      ext.name,
      ext.description,
      ext.weight,
      ext.price,
      ext.main_category,
      ext.sub_category
FROM master_product_clusters mpc
JOIN external_temp_products_pareto ext ON mpc.product_source_id = ext.id_source
WHERE mpc.master_product_status_id = 2
"""


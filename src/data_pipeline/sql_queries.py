FACT_ORDERS_SQL = """
WITH payment_summary AS (
    SELECT
        order_id,
        SUM(payment_value) AS payment_value,
        AVG(payment_installments) AS avg_installments,
        MAX(payment_type) AS primary_payment_type
    FROM order_payments
    GROUP BY order_id
),
item_summary AS (
    SELECT
        oi.order_id,
        COUNT(*) AS item_count,
        COUNT(DISTINCT oi.product_id) AS distinct_product_count,
        COUNT(DISTINCT oi.seller_id) AS seller_count,
        SUM(oi.price) AS item_revenue,
        SUM(oi.freight_value) AS freight_value,
        MAX(COALESCE(t.product_category_name_english, p.product_category_name)) AS product_category
    FROM order_items oi
    LEFT JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN category_translation t ON p.product_category_name = t.product_category_name
    GROUP BY oi.order_id
)
SELECT
    o.order_id,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    o.order_status,
    CAST(o.order_purchase_timestamp AS timestamp) AS order_purchase_ts,
    CAST(o.order_delivered_customer_date AS timestamp) AS delivered_customer_ts,
    CAST(o.order_estimated_delivery_date AS timestamp) AS estimated_delivery_ts,
    COALESCE(i.item_count, 0) AS item_count,
    COALESCE(i.distinct_product_count, 0) AS distinct_product_count,
    COALESCE(i.seller_count, 0) AS seller_count,
    COALESCE(i.item_revenue, 0.0) AS item_revenue,
    COALESCE(i.freight_value, 0.0) AS freight_value,
    COALESCE(p.payment_value, i.item_revenue + i.freight_value, 0.0) AS payment_value,
    COALESCE(p.avg_installments, 1.0) AS avg_installments,
    COALESCE(p.primary_payment_type, 'unknown') AS primary_payment_type,
    COALESCE(i.product_category, 'unknown') AS product_category,
    CASE
        WHEN o.order_delivered_customer_date IS NOT NULL
         AND o.order_estimated_delivery_date IS NOT NULL
         AND CAST(o.order_delivered_customer_date AS timestamp) > CAST(o.order_estimated_delivery_date AS timestamp)
        THEN 1 ELSE 0
    END AS is_late_delivery
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN item_summary i ON o.order_id = i.order_id
LEFT JOIN payment_summary p ON o.order_id = p.order_id
WHERE o.order_purchase_timestamp IS NOT NULL
"""

CUSTOMER_FEATURES_SQL = """
WITH max_date AS (
    SELECT MAX(order_purchase_ts) AS snapshot_date FROM fact_orders
),
customer_base AS (
    SELECT
        customer_unique_id,
        COUNT(DISTINCT order_id) AS order_count,
        SUM(payment_value) AS total_revenue,
        AVG(payment_value) AS avg_order_value,
        SUM(item_count) AS total_items,
        AVG(is_late_delivery) AS late_delivery_rate,
        MIN(order_purchase_ts) AS first_purchase_ts,
        MAX(order_purchase_ts) AS last_purchase_ts,
        COUNT(DISTINCT product_category) AS category_count
    FROM fact_orders
    WHERE order_status IN ('delivered', 'shipped', 'invoiced', 'processing', 'approved')
    GROUP BY customer_unique_id
)
SELECT
    cb.*,
    DATEDIFF(md.snapshot_date, cb.last_purchase_ts) AS recency_days,
    DATEDIFF(cb.last_purchase_ts, cb.first_purchase_ts) AS tenure_days,
    CASE WHEN DATEDIFF(md.snapshot_date, cb.last_purchase_ts) > 120 THEN 1 ELSE 0 END AS churn_label
FROM customer_base cb
CROSS JOIN max_date md
"""

CATEGORY_DAILY_SALES_SQL = """
SELECT
    TO_DATE(order_purchase_ts) AS order_date,
    product_category,
    COUNT(DISTINCT order_id) AS orders,
    SUM(payment_value) AS revenue
FROM fact_orders
WHERE order_status IN ('delivered', 'shipped', 'invoiced', 'processing', 'approved')
GROUP BY TO_DATE(order_purchase_ts), product_category
"""

STATE_CATEGORY_DAILY_SALES_SQL = """
SELECT
    TO_DATE(order_purchase_ts) AS order_date,
    customer_state,
    product_category,
    COUNT(DISTINCT order_id) AS orders,
    SUM(payment_value) AS revenue
FROM fact_orders
WHERE order_status IN ('delivered', 'shipped', 'invoiced', 'processing', 'approved')
GROUP BY TO_DATE(order_purchase_ts), customer_state, product_category
"""

CUSTOMER_REPEAT_FEATURES_SQL = """
WITH ranked_orders AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY customer_unique_id ORDER BY order_purchase_ts) AS order_rank,
        COUNT(DISTINCT order_id) OVER (PARTITION BY customer_unique_id) AS lifetime_orders
    FROM fact_orders
    WHERE order_status IN ('delivered', 'shipped', 'invoiced', 'processing', 'approved')
)
SELECT
    customer_unique_id,
    customer_state AS first_customer_state,
    product_category AS first_product_category,
    primary_payment_type AS first_payment_type,
    payment_value AS first_order_value,
    item_count AS first_item_count,
    freight_value AS first_freight_value,
    avg_installments AS first_avg_installments,
    is_late_delivery AS first_late_delivery,
    DATEDIFF(delivered_customer_ts, order_purchase_ts) AS first_delivery_days,
    MONTH(order_purchase_ts) AS first_order_month,
    CASE WHEN lifetime_orders > 1 THEN 1 ELSE 0 END AS repeat_purchase_label
FROM ranked_orders
WHERE order_rank = 1
"""

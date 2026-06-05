-- Thay {THRESHOLD} bằng giá trị (e.g., 70)

SELECT 
    cms_code_enc,
    ROUND(churn_rate::numeric, 2) as churn_rate,
    ROUND(model_probability_pct::numeric, 6) as model_probability_pct,
    item_last,
    ROUND(revenue_last::numeric/1000000, 2) as revenue_m,
    complaint_last,
    reason_1,
    created_at
FROM data_static.cus_risk_{THRESHOLD}
ORDER BY churn_rate DESC
LIMIT 20;

-- Check dữ liệu quality
SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT cms_code_enc) as unique_customers,
    COUNT(CASE WHEN churn_rate IS NULL THEN 1 END) as null_churn_rates,
    ROUND(MIN(churn_rate)::numeric, 2) as min_rate,
    ROUND(MAX(churn_rate)::numeric, 2) as max_rate,
    ROUND(AVG(churn_rate)::numeric, 2) as avg_rate
FROM data_static.cus_risk_{THRESHOLD};

-- Check for duplicates
SELECT cms_code_enc, COUNT(*) as dup_count
FROM data_static.cus_risk_{THRESHOLD}
GROUP BY cms_code_enc
HAVING COUNT(*) > 1;

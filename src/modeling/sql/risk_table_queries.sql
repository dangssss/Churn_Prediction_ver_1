-- ============================================================================
-- SQL Template: Risk Table Queries & Analysis
-- ============================================================================
-- Các query hữu ích để làm việc với bảng risk sau khi 03_export_table_risk.py chạy xong

-- ============================================================================
-- 1. VIEW TOP 20 CUSTOMERS AT HIGHEST RISK
-- ============================================================================

SELECT 
    cms_code_enc,
    ROUND(churn_rate::numeric, 2) as churn_rate,
    item_last,
    revenue_last::numeric/1000000 as revenue_last_m,
    complaint_last,
    delay_last,
    nodone_last,
    reason_1,
    reason_2,
    reason_3,
    created_at
FROM data_static.cus_risk_70
ORDER BY churn_rate DESC, cms_code_enc
LIMIT 20;


-- ============================================================================
-- 2. DISTRIBUTION OF CHURN RISK
-- ============================================================================

SELECT 
    CASE 
        WHEN churn_rate >= 90 THEN '90-100%'
        WHEN churn_rate >= 80 THEN '80-90%'
        WHEN churn_rate >= 70 THEN '70-80%'
        ELSE '<70%'
    END as risk_bucket,
    COUNT(*) as num_customers,
    ROUND(AVG(churn_rate)::numeric, 2) as avg_churn_rate,
    MIN(churn_rate)::numeric as min_rate,
    MAX(churn_rate)::numeric as max_rate
FROM data_static.cus_risk_70
GROUP BY risk_bucket
ORDER BY min_rate DESC;


-- ============================================================================
-- 3. TOP REASONS FOR CHURN
-- ============================================================================

-- Frequency of reason_1
SELECT 
    reason_1,
    COUNT(*) as frequency,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM data_static.cus_risk_70)::numeric, 2) as pct
FROM data_static.cus_risk_70
WHERE reason_1 IS NOT NULL
GROUP BY reason_1
ORDER BY frequency DESC;

-- Combine all reasons (unnest)
-- SELECT 
--     reason,
--     COUNT(*) as frequency
-- FROM (
--     SELECT reason_1 as reason FROM data_static.cus_risk_70 WHERE reason_1 IS NOT NULL
--     UNION ALL
--     SELECT reason_2 FROM data_static.cus_risk_70 WHERE reason_2 IS NOT NULL
--     UNION ALL
--     SELECT reason_3 FROM data_static.cus_risk_70 WHERE reason_3 IS NOT NULL
-- ) t
-- GROUP BY reason
-- ORDER BY frequency DESC;


-- ============================================================================
-- 4. CUSTOMERS WITH SPECIFIC REASON
-- ============================================================================

-- Find customers with "Số đơn giảm mạnh" as primary reason
SELECT 
    cms_code_enc,
    churn_rate,
    item_last,
    reason_1,
    reason_2,
    reason_3
FROM data_static.cus_risk_70
WHERE reason_1 = 'Số đơn giảm mạnh'
ORDER BY churn_rate DESC
LIMIT 20;


-- ============================================================================
-- 5. EXPORT RISK DATA FOR CAMPAIGN
-- ============================================================================

-- Export to CSV format (for Retention Campaign)
COPY (
    SELECT 
        cms_code_enc,
        ROUND(churn_rate::numeric, 2) as churn_rate,
        item_last,
        ROUND(revenue_last::numeric/1000000, 2) as revenue_last_m,
        complaint_last,
        reason_1,
        reason_2,
        reason_3
    FROM data_static.cus_risk_70
    WHERE churn_rate >= 75  -- High risk only
    ORDER BY churn_rate DESC
) TO STDOUT WITH CSV HEADER;


-- ============================================================================
-- 6. STATISTICAL SUMMARY
-- ============================================================================

SELECT 
    'Total Customers' as metric,
    COUNT(*)::text as value
FROM data_static.cus_risk_70
UNION ALL
SELECT 'Avg Churn Rate (%)', ROUND(AVG(churn_rate)::numeric, 2)::text
FROM data_static.cus_risk_70
UNION ALL
SELECT 'Median Churn Rate (%)', ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY churn_rate)::numeric, 2)::text
FROM data_static.cus_risk_70
UNION ALL
SELECT 'Max Churn Rate (%)', ROUND(MAX(churn_rate)::numeric, 2)::text
FROM data_static.cus_risk_70
UNION ALL
SELECT 'Min Churn Rate (%)', ROUND(MIN(churn_rate)::numeric, 2)::text
FROM data_static.cus_risk_70
UNION ALL
SELECT 'Std Dev Churn Rate', ROUND(STDDEV(churn_rate)::numeric, 2)::text
FROM data_static.cus_risk_70
UNION ALL
SELECT 'Customers with reason_1', COUNT(*)::text
FROM data_static.cus_risk_70
WHERE reason_1 IS NOT NULL
UNION ALL
SELECT 'Customers with reason_2', COUNT(*)::text
FROM data_static.cus_risk_70
WHERE reason_2 IS NOT NULL
UNION ALL
SELECT 'Customers with reason_3', COUNT(*)::text
FROM data_static.cus_risk_70
WHERE reason_3 IS NOT NULL;


-- ============================================================================
-- 7. MULTIPLE THRESHOLDS COMPARISON
-- ============================================================================

-- If you have multiple risk tables (70%, 75%, 80%, etc.)
SELECT 
    '70% Threshold' as threshold,
    COUNT(*) as num_customers
FROM data_static.cus_risk_70
UNION ALL
SELECT '75% Threshold', COUNT(*)
FROM data_static.cus_risk_75
UNION ALL
SELECT '80% Threshold', COUNT(*)
FROM data_static.cus_risk_80;


-- ============================================================================
-- 8. IDENTIFY FALSE POSITIVES/NEGATIVES
-- ============================================================================

-- Join with actual churn labels (if you have them)
-- Assuming you have a ground_truth table with cms_code_enc and actual_churn
SELECT 
    CASE 
        WHEN gr.actual_churn = 1 AND cr.cms_code_enc IS NOT NULL THEN 'True Positive'
        WHEN gr.actual_churn = 1 AND cr.cms_code_enc IS NULL THEN 'False Negative'
        WHEN gr.actual_churn = 0 AND cr.cms_code_enc IS NOT NULL THEN 'False Positive'
        ELSE 'True Negative'
    END as prediction_type,
    COUNT(*) as count
FROM ground_truth gr
LEFT JOIN data_static.cus_risk_70 cr ON gr.cms_code_enc = cr.cms_code_enc
GROUP BY prediction_type
ORDER BY count DESC;


-- ============================================================================
-- 9. CLEAN UP OLD RISK TABLES (if needed)
-- ============================================================================

-- List all risk tables
SELECT 
    table_name,
    (SELECT COUNT(*) FROM information_schema.tables t 
     WHERE t.table_schema = 'data_static' AND t.table_name = 'cus_risk_' || SUBSTRING(table_name, 10)) as row_count
FROM information_schema.tables
WHERE table_schema = 'data_static' AND table_name LIKE 'cus_risk_%'
ORDER BY table_name;

-- Drop old risk table (if needed)
-- DROP TABLE IF EXISTS data_static.cus_risk_70 CASCADE;


-- ============================================================================
-- 10. ADVANCED: CREATE MATERIALIZED VIEW
-- ============================================================================

-- Create a view for easier access
CREATE OR REPLACE VIEW v_risk_summary AS
SELECT 
    cms_code_enc,
    churn_rate,
    CASE 
        WHEN churn_rate >= 90 THEN 'Critical'
        WHEN churn_rate >= 80 THEN 'High'
        WHEN churn_rate >= 70 THEN 'Medium'
        ELSE 'Low'
    END as risk_level,
    item_last,
    revenue_last,
    complaint_last,
    delay_last,
    nodone_last,
    order_score_last,
    satisfation_last,
    reason_1,
    reason_2,
    reason_3,
    created_at
FROM data_static.cus_risk_70;

-- Query via view
SELECT * FROM v_risk_summary 
WHERE risk_level IN ('Critical', 'High')
ORDER BY churn_rate DESC
LIMIT 30;

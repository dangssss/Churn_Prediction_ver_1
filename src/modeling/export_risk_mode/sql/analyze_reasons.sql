-- Thay {THRESHOLD} bằng giá trị (e.g., 70)

-- Frequency of reason_1
SELECT 
    reason_1,
    COUNT(*) as frequency,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM data_static.cus_risk_{THRESHOLD})::numeric, 2) as pct
FROM data_static.cus_risk_{THRESHOLD}
WHERE reason_1 IS NOT NULL
GROUP BY reason_1
ORDER BY frequency DESC;

-- Frequency of all reasons combined
SELECT 
    reason,
    COUNT(*) as frequency,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM data_static.cus_risk_{THRESHOLD})::numeric, 2) as pct
FROM (
    SELECT reason_1 as reason FROM data_static.cus_risk_{THRESHOLD} WHERE reason_1 IS NOT NULL
    UNION ALL
    SELECT reason_2 FROM data_static.cus_risk_{THRESHOLD} WHERE reason_2 IS NOT NULL
    UNION ALL
    SELECT reason_3 FROM data_static.cus_risk_{THRESHOLD} WHERE reason_3 IS NOT NULL
) t
GROUP BY reason
ORDER BY frequency DESC;

-- Customers with specific reason
SELECT 
    cms_code_enc,
    churn_rate,
    item_last,
    reason_1,
    reason_2,
    reason_3
FROM data_static.cus_risk_{THRESHOLD}
WHERE reason_1 = 'Số đơn giảm mạnh'
ORDER BY churn_rate DESC
LIMIT 20;

-- Check reason distribution
SELECT 
    SUM(CASE WHEN reason_1 IS NOT NULL THEN 1 ELSE 0 END) as with_reason_1,
    SUM(CASE WHEN reason_2 IS NOT NULL THEN 1 ELSE 0 END) as with_reason_2,
    SUM(CASE WHEN reason_3 IS NOT NULL THEN 1 ELSE 0 END) as with_reason_3,
    SUM(CASE WHEN reason_1 IS NULL THEN 1 ELSE 0 END) as without_reasons,
    COUNT(*) as total
FROM data_static.cus_risk_{THRESHOLD};

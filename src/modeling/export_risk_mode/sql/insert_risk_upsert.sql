-- UPSERT from temp table to main risk table
INSERT INTO data_static.{TABLE_NAME}
SELECT * FROM {TEMP_TABLE}
ON CONFLICT (cms_code_enc, predict_period) DO UPDATE SET
    window_end = EXCLUDED.window_end,
    item_last = EXCLUDED.item_last,
    revenue_last = EXCLUDED.revenue_last,
    complaint_last = EXCLUDED.complaint_last,
    delay_last = EXCLUDED.delay_last,
    nodone_last = EXCLUDED.nodone_last,
    order_score_last = EXCLUDED.order_score_last,
    satisfaction_last = EXCLUDED.satisfaction_last,
    churn_rate = EXCLUDED.churn_rate,
    model_probability_pct = EXCLUDED.model_probability_pct,
    reason_1 = EXCLUDED.reason_1,
    reason_2 = EXCLUDED.reason_2,
    reason_3 = EXCLUDED.reason_3,
    update_at = now();

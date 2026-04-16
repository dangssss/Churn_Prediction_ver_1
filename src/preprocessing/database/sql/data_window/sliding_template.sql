-- Placeholders: {TABLE_NAME}, {TABLE_SAFE}
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    cms_code_enc VARCHAR(64),
    window_size INT,
    window_start VARCHAR(6),
    window_end VARCHAR(6),
    PRIMARY KEY (cms_code_enc, window_size, window_start, window_end),

    -- Monthly metrics (item, revenue, complaint, delay, nodone, order_score, satisfaction per month)
    {MONTHLY_COLUMNS},

    -- Volume Metrics (aggregate across window)
    item_sum BIGINT,
    item_avg DOUBLE PRECISION,
    item_std DOUBLE PRECISION,
    item_min_month BIGINT,
    item_max_month BIGINT,
    item_median DOUBLE PRECISION,

    -- Revenue Metrics
    revenue_sum BIGINT,
    revenue_avg DOUBLE PRECISION,
    revenue_std DOUBLE PRECISION,
    revenue_min_month BIGINT,
    revenue_max_month BIGINT,
    revenue_median DOUBLE PRECISION,

    -- Complaint Metrics
    complaint_sum BIGINT,
    complaint_avg DOUBLE PRECISION,
    complaint_std DOUBLE PRECISION,
    complaint_min_month BIGINT,
    complaint_max_month BIGINT,
    complaint_median DOUBLE PRECISION,
    complaint_diversity INT,
    most_common_complaint INT,

    -- Weight Metrics
    weight_sum DOUBLE PRECISION,
    weight_avg DOUBLE PRECISION,
    weight_std DOUBLE PRECISION,

    avg_revenue_per_item DOUBLE PRECISION,
    avg_weight_per_item DOUBLE PRECISION,

    -- Rates & ratios
    pct_delay DOUBLE PRECISION,
    pct_refund DOUBLE PRECISION,
    pct_noaccepted DOUBLE PRECISION,
    pct_lost_order DOUBLE PRECISION,
    pct_complaint DOUBLE PRECISION,
    pct_complaint_per_item DOUBLE PRECISION,
    pct_successful_item DOUBLE PRECISION,
    pct_intra_province DOUBLE PRECISION,
    pct_international DOUBLE PRECISION,
    avg_delayday DOUBLE PRECISION,

    -- Quality scores
    order_score_avg DOUBLE PRECISION,
    order_score_std DOUBLE PRECISION,
    order_min_month DOUBLE PRECISION,
    order_max_month DOUBLE PRECISION,
    satisfaction_avg DOUBLE PRECISION,
    satisfaction_std DOUBLE PRECISION,
    satisfaction_min_month DOUBLE PRECISION,
    satisfaction_max_month DOUBLE PRECISION,

    -- Activity pattern
    active_months INT,
    inactive_months INT,
    active_days INT,
    inactive_days INT,
    avg_noservice_days DOUBLE PRECISION,
    max_consecutive_inactive INT,
    avg_lastday DOUBLE PRECISION,

    -- Slope features
    item_slope DOUBLE PRECISION,
    revenue_slope DOUBLE PRECISION,
    satisfy_slope DOUBLE PRECISION,
    complaint_slope DOUBLE PRECISION,

    -- Volatility
    cv_item DOUBLE PRECISION,
    cv_revenue DOUBLE PRECISION,
    item_range BIGINT,
    revenue_range BIGINT,

    -- Service mix
    service_types_used INT,
    dominant_service VARCHAR(10),
    dominant_service_ratio DOUBLE PRECISION,

    ser_c_sum INT,
    ser_e_sum INT,
    ser_m_sum INT,
    ser_p_sum INT,
    ser_r_sum INT,
    ser_u_sum INT,
    ser_l_sum INT,
    ser_q_sum INT,

    -- RFM
    recency INT,
    frequency DOUBLE PRECISION,
    monetary DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_wt_window_start_{TABLE_SAFE} ON {TABLE_NAME} (window_start);
CREATE INDEX IF NOT EXISTS idx_wt_window_end_{TABLE_SAFE} ON {TABLE_NAME} (window_end);

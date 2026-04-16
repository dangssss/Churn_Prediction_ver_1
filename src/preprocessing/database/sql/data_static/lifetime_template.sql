CREATE TABLE IF NOT EXISTS data_static.cus_lifetime (
    cms_code_enc VARCHAR(64),

    -- Contract info
    is_corporate BOOLEAN,
    contract_classify INT,
    contract_service INT,
    custype INT,
    contract_sig_first DATE,
    tenure INT,
    contract_mgr_org INT,
    cus_poscode INT,
    cus_province INT,

    -- Lifetime aggregations
    lifetime_total_items BIGINT,
    lifetime_total_revenue BIGINT,
    lifetime_total_weight DECIMAL(12,3),
    lifetime_total_complaint BIGINT,
    lifetime_avg_revenue_per_item DECIMAL(12,3),
    lifetime_avg_weight_per_item DECIMAL(12,3),
    lifetime_months_active INT,
    lifetime_days_active INT,

    -- Quality metrics
    lifetime_pct_delay DECIMAL(10,3),
    lifetime_pct_refund DECIMAL(10,3),
    lifetime_pct_noaccepted DECIMAL(10,3),
    lifetime_pct_lost_order DECIMAL(10,3),
    lifetime_pct_complaint DECIMAL(10,3),
    lifetime_pct_complaint_per_item DECIMAL(10,3),
    lifetime_pct_successful_item DECIMAL(10,3),
    lifetime_avg_delayday DECIMAL(10,3),
    lifetime_avg_order_score DECIMAL(10,3),
    lifetime_avg_satisfaction DECIMAL(10,3),

    -- Service usage
    lifetime_service_types_count INT,
    lifetime_dominant_service VARCHAR(10),
    lifetime_pct_international DECIMAL(10,3),
    lifetime_pct_intra_province DECIMAL(10,3),

    -- Geographic
    most_common_rec_province INT,
    most_common_rec_district INT,
    most_common_rec_commune INT,
    most_common_region VARCHAR(20),
    
    PRIMARY KEY (cms_code_enc)
);

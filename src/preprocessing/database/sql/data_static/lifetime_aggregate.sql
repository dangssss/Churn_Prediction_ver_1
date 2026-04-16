CREATE INDEX IF NOT EXISTS idx_cas_customer_code_month ON public.cas_customer(cms_code_enc, report_month);
CREATE INDEX IF NOT EXISTS idx_cas_customer_month ON public.cas_customer(report_month);
CREATE INDEX IF NOT EXISTS idx_cas_info_code ON public.cas_info(cms_code_enc);
CREATE INDEX IF NOT EXISTS idx_cms_complaint_code_date ON public.cms_complaint(cms_code_enc, create_complaint_date);
CREATE INDEX IF NOT EXISTS idx_cms_complaint_date ON public.cms_complaint(create_complaint_date);

WITH customer_agg AS (
    SELECT
        cms_code_enc,
        -- Volume metrics
        SUM(item_count)::bigint AS lifetime_total_items,
        SUM(total_fee)::bigint AS lifetime_total_revenue,
        SUM(weight_kg)::double precision AS lifetime_total_weight,
        SUM(total_complaint)::bigint AS lifetime_total_complaint,
        -- Per-item ratios (computed inline to avoid re-scanning)
        COALESCE(SUM(total_fee)::double precision / NULLIF(SUM(item_count),0), 0) AS lifetime_avg_revenue_per_item,
        COALESCE(SUM(weight_kg)::double precision / NULLIF(SUM(item_count),0), 0) AS lifetime_avg_weight_per_item,
        -- Activity
        COUNT(report_month) AS lifetime_months_active,
        MIN(report_month) AS first_month,
        -- Quality metrics (inline to avoid second scan)
        COALESCE(SUM(delay_count)::double precision / NULLIF(SUM(item_count),0), 0) AS lifetime_pct_delay,
        COALESCE(SUM(refunded)::double precision / NULLIF(SUM(item_count),0), 0) AS lifetime_pct_refund,
        COALESCE(SUM(noaccepted)::double precision / NULLIF(SUM(item_count),0), 0) AS lifetime_pct_noaccepted,
        COALESCE(SUM(lost_order)::double precision / NULLIF(SUM(item_count),0), 0) AS lifetime_pct_lost_order,
        COALESCE(SUM(total_complaint)::double precision / NULLIF(SUM(item_count),0), 0) AS lifetime_pct_complaint,
        COALESCE(1 - SUM(nodone)::double precision / NULLIF(SUM(item_count),0), 0) AS lifetime_pct_successful_item,
        COALESCE(SUM(CASE WHEN delay_count > 0 THEN delay_day ELSE 0 END)::double precision / NULLIF(SUM(CASE WHEN delay_count > 0 THEN 1 ELSE 0 END), 0), 0) AS lifetime_avg_delayday,
        COALESCE(AVG(order_score), 0)::double precision AS lifetime_avg_order_score,
        COALESCE(AVG(satisfaction_score), 0)::double precision AS lifetime_avg_satisfaction,
        COALESCE(SUM(international)::double precision / NULLIF(SUM(item_count),0), 0) AS lifetime_pct_international,
        COALESCE(SUM(CASE WHEN intra_province = 1 THEN 1 ELSE 0 END)::double precision / NULLIF(SUM(item_count),0), 0) AS lifetime_pct_intra_province,
        -- Service usage (summed here, dominant picked in SELECT)
        SUM(COALESCE(ser_c,0))::bigint AS ser_c,
        SUM(COALESCE(ser_e,0))::bigint AS ser_e,
        SUM(COALESCE(ser_m,0))::bigint AS ser_m,
        SUM(COALESCE(ser_p,0))::bigint AS ser_p,
        SUM(COALESCE(ser_r,0))::bigint AS ser_r,
        SUM(COALESCE(ser_u,0))::bigint AS ser_u,
        SUM(COALESCE(ser_l,0))::bigint AS ser_l,
        SUM(COALESCE(ser_q,0))::bigint AS ser_q
    FROM public.cas_customer
    WHERE report_month >= DATE '{START_DATE}' 
    GROUP BY cms_code_enc
),

info AS (
    SELECT DISTINCT ON (cms_code_enc)
        cms_code_enc,
        COALESCE(contract_classify, 1) AS contract_classify,
        COALESCE(contract_service, 62) AS contract_service,
        COALESCE(custype, 1) AS custype,
        contract_sig_first,
        tenure,
        contract_mgr_org,
        cus_poscode,
        cus_province
    FROM public.cas_info
    ORDER BY cms_code_enc, contract_sig_first DESC NULLS LAST
),

bccp_agg AS (
    SELECT
        cms_code_enc,
        COUNT(DISTINCT DATE(sending_time))::int AS lifetime_days_active,
        COALESCE(SUM(CASE WHEN total_complaint > 0 THEN 1 ELSE 0 END)::double precision / NULLIF(COUNT(*), 0), 0) AS lifetime_pct_complaint_per_item,
        (MODE() WITHIN GROUP (ORDER BY rec_province_code))::bigint AS most_common_rec_province,
        (MODE() WITHIN GROUP (ORDER BY rec_district_code))::bigint AS most_common_rec_district,
        (MODE() WITHIN GROUP (ORDER BY rec_commune_code))::bigint AS most_common_rec_commune,
        (MODE() WITHIN GROUP (ORDER BY region))::varchar AS most_common_region
    FROM {BCCP_SRC}
    WHERE sending_time >= DATE '{START_DATE}'
    GROUP BY cms_code_enc
)

INSERT INTO data_static.cus_lifetime (
    cms_code_enc, is_corporate,
    contract_classify, contract_service, custype, contract_sig_first, tenure, contract_mgr_org, cus_poscode, cus_province,
    lifetime_total_items, lifetime_total_revenue, lifetime_total_weight, lifetime_total_complaint,
    lifetime_avg_revenue_per_item, lifetime_avg_weight_per_item, lifetime_months_active, lifetime_days_active,
    lifetime_pct_delay, lifetime_pct_refund, lifetime_pct_noaccepted, lifetime_pct_lost_order, lifetime_pct_complaint,
    lifetime_pct_complaint_per_item, lifetime_pct_successful_item, lifetime_avg_delayday,
    lifetime_avg_order_score, lifetime_avg_satisfaction,
    lifetime_service_types_count, lifetime_dominant_service,
    lifetime_pct_international, lifetime_pct_intra_province,
    most_common_rec_province, most_common_rec_district, most_common_rec_commune, most_common_region)
SELECT
    ca.cms_code_enc,
    (LEFT(ca.cms_code_enc, 1) = 'T') AS is_corporate,
    COALESCE(i.contract_classify, 1),
    COALESCE(i.contract_service, 62),
    COALESCE(i.custype, 1),
    COALESCE(i.contract_sig_first, date_trunc('month', ca.first_month))::timestamp,
    COALESCE(i.tenure,
        (EXTRACT(year from age(now(), COALESCE(i.contract_sig_first, date_trunc('month', ca.first_month))))*12 + 
         EXTRACT(month from age(now(), COALESCE(i.contract_sig_first, date_trunc('month', ca.first_month)))))::int
    ),
    i.contract_mgr_org,
    i.cus_poscode,
    i.cus_province,
    ca.lifetime_total_items,
    ca.lifetime_total_revenue,
    ca.lifetime_total_weight,
    ca.lifetime_total_complaint,
    ca.lifetime_avg_revenue_per_item,
    ca.lifetime_avg_weight_per_item,
    ca.lifetime_months_active,
    COALESCE(b.lifetime_days_active, 0)::int,
    ca.lifetime_pct_delay,
    ca.lifetime_pct_refund,
    ca.lifetime_pct_noaccepted,
    ca.lifetime_pct_lost_order,
    ca.lifetime_pct_complaint,
    COALESCE(b.lifetime_pct_complaint_per_item, 0),
    ca.lifetime_pct_successful_item,
    ca.lifetime_avg_delayday,
    ca.lifetime_avg_order_score,
    ca.lifetime_avg_satisfaction,
    -- Service types count
    ((CASE WHEN ca.ser_c > 0 THEN 1 ELSE 0 END) + (CASE WHEN ca.ser_e > 0 THEN 1 ELSE 0 END) + 
     (CASE WHEN ca.ser_m > 0 THEN 1 ELSE 0 END) + (CASE WHEN ca.ser_p > 0 THEN 1 ELSE 0 END) + 
     (CASE WHEN ca.ser_r > 0 THEN 1 ELSE 0 END) + (CASE WHEN ca.ser_u > 0 THEN 1 ELSE 0 END) + 
     (CASE WHEN ca.ser_l > 0 THEN 1 ELSE 0 END) + (CASE WHEN ca.ser_q > 0 THEN 1 ELSE 0 END))::int,
    -- Dominant service (simplified without complex ARRAY indexing)
    CASE 
        WHEN GREATEST(ca.ser_c, ca.ser_e, ca.ser_m, ca.ser_p, ca.ser_r, ca.ser_u, ca.ser_l, ca.ser_q) = 0 THEN 'U'
        WHEN GREATEST(ca.ser_c, ca.ser_e, ca.ser_m, ca.ser_p, ca.ser_r, ca.ser_u, ca.ser_l, ca.ser_q) = ca.ser_c THEN 'C'
        WHEN GREATEST(ca.ser_c, ca.ser_e, ca.ser_m, ca.ser_p, ca.ser_r, ca.ser_u, ca.ser_l, ca.ser_q) = ca.ser_e THEN 'E'
        WHEN GREATEST(ca.ser_c, ca.ser_e, ca.ser_m, ca.ser_p, ca.ser_r, ca.ser_u, ca.ser_l, ca.ser_q) = ca.ser_m THEN 'M'
        WHEN GREATEST(ca.ser_c, ca.ser_e, ca.ser_m, ca.ser_p, ca.ser_r, ca.ser_u, ca.ser_l, ca.ser_q) = ca.ser_p THEN 'P'
        WHEN GREATEST(ca.ser_c, ca.ser_e, ca.ser_m, ca.ser_p, ca.ser_r, ca.ser_u, ca.ser_l, ca.ser_q) = ca.ser_r THEN 'R'
        WHEN GREATEST(ca.ser_c, ca.ser_e, ca.ser_m, ca.ser_p, ca.ser_r, ca.ser_u, ca.ser_l, ca.ser_q) = ca.ser_u THEN 'U'
        WHEN GREATEST(ca.ser_c, ca.ser_e, ca.ser_m, ca.ser_p, ca.ser_r, ca.ser_u, ca.ser_l, ca.ser_q) = ca.ser_l THEN 'L'
        WHEN GREATEST(ca.ser_c, ca.ser_e, ca.ser_m, ca.ser_p, ca.ser_r, ca.ser_u, ca.ser_l, ca.ser_q) = ca.ser_q THEN 'Q'
        ELSE 'U'
    END,
    ca.lifetime_pct_international,
    ca.lifetime_pct_intra_province,
    COALESCE(b.most_common_rec_province, 0),
    COALESCE(b.most_common_rec_district, 0),
    COALESCE(b.most_common_rec_commune, 0),
    COALESCE(b.most_common_region, 'N/A')
FROM customer_agg ca
LEFT JOIN info i ON i.cms_code_enc = ca.cms_code_enc
LEFT JOIN bccp_agg b ON b.cms_code_enc = ca.cms_code_enc
ON CONFLICT (cms_code_enc) DO UPDATE SET
    contract_service = EXCLUDED.contract_service,
    custype = EXCLUDED.custype,
    contract_sig_first = EXCLUDED.contract_sig_first,
    tenure = EXCLUDED.tenure,
    contract_mgr_org = EXCLUDED.contract_mgr_org,
    cus_poscode = EXCLUDED.cus_poscode,
    cus_province = EXCLUDED.cus_province,
    lifetime_total_items = EXCLUDED.lifetime_total_items,
    lifetime_total_revenue = EXCLUDED.lifetime_total_revenue,
    lifetime_total_weight = EXCLUDED.lifetime_total_weight,
    lifetime_total_complaint = EXCLUDED.lifetime_total_complaint,
    lifetime_avg_revenue_per_item = EXCLUDED.lifetime_avg_revenue_per_item,
    lifetime_avg_weight_per_item = EXCLUDED.lifetime_avg_weight_per_item,
    lifetime_months_active = EXCLUDED.lifetime_months_active,
    lifetime_days_active = EXCLUDED.lifetime_days_active,
    lifetime_pct_delay = EXCLUDED.lifetime_pct_delay,
    lifetime_pct_refund = EXCLUDED.lifetime_pct_refund,
    lifetime_pct_noaccepted = EXCLUDED.lifetime_pct_noaccepted,
    lifetime_pct_lost_order = EXCLUDED.lifetime_pct_lost_order,
    lifetime_pct_complaint = EXCLUDED.lifetime_pct_complaint,
    lifetime_pct_complaint_per_item = EXCLUDED.lifetime_pct_complaint_per_item,
    lifetime_pct_successful_item = EXCLUDED.lifetime_pct_successful_item,
    lifetime_avg_delayday = EXCLUDED.lifetime_avg_delayday,
    lifetime_avg_order_score = EXCLUDED.lifetime_avg_order_score,
    lifetime_avg_satisfaction = EXCLUDED.lifetime_avg_satisfaction,
    lifetime_service_types_count = EXCLUDED.lifetime_service_types_count,
    lifetime_dominant_service = EXCLUDED.lifetime_dominant_service,
    lifetime_pct_international = EXCLUDED.lifetime_pct_international,
    lifetime_pct_intra_province = EXCLUDED.lifetime_pct_intra_province,
    most_common_rec_province = EXCLUDED.most_common_rec_province,
    most_common_rec_district = EXCLUDED.most_common_rec_district,
    most_common_rec_commune = EXCLUDED.most_common_rec_commune,
    most_common_region = EXCLUDED.most_common_region;

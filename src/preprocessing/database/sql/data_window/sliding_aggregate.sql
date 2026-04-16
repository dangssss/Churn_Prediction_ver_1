WITH cms AS (
    SELECT *
    FROM {WINDOW_SOURCE_TABLE}
    WHERE report_month >= DATE '{START_DATE}' AND report_month <= DATE '{END_DATE}'
),

monthly_metrics AS (
    SELECT
        cms_code_enc,
        to_char(report_month, 'YYMM') AS month_key,
        to_char(report_month, 'YYMM')::bigint AS month_key_num,
        report_month,
        item_count,
        total_fee,
        total_complaint,
        delay_count,
        delay_day,
        nodone,
        refunded,
        noaccepted,
        lost_order,
        intra_province,
        international,
        order_score,
        satisfaction_score,
        weight_kg,
        lastday,
        ser_c, ser_e, ser_m, ser_p, ser_r, ser_u, ser_l, ser_q
    FROM cms
),

monthly_sums AS (
    SELECT
        cms_code_enc,
        month_key,
        month_key_num,
        SUM(item_count)::bigint AS item_sum,
        SUM(total_fee)::bigint AS revenue_sum,
        SUM(total_complaint)::bigint AS complaint_sum,
        SUM(delay_count)::bigint AS delay_sum,
        SUM(nodone)::bigint AS nodone_sum,
        AVG(order_score)::double precision AS order_score_avg,
        AVG(satisfaction_score)::double precision AS satisfaction_avg,
        COUNT(*)::int AS month_record_count,
        COUNT(CASE WHEN total_complaint > 0 THEN 1 END)::int AS months_with_complaint
    FROM monthly_metrics
    GROUP BY cms_code_enc, month_key, month_key_num
),

monthly_pivoted AS (
    SELECT
        cms_code_enc, {MONTHLY_CASE_STATEMENTS}
    FROM monthly_sums
    GROUP BY cms_code_enc
),

month_extremes AS (
    SELECT
        cms_code_enc,
        MAX(CASE WHEN item_count_rank = 1 THEN month_key_num END) AS item_max_month,
        MAX(CASE WHEN item_count_rank_desc = 1 THEN month_key_num END) AS item_min_month,
        MAX(CASE WHEN revenue_rank = 1 THEN month_key_num END) AS revenue_max_month,
        MAX(CASE WHEN revenue_rank_desc = 1 THEN month_key_num END) AS revenue_min_month,
        MAX(CASE WHEN complaint_rank = 1 AND month_key_num IS NOT NULL THEN month_key_num END) AS complaint_max_month,
        MAX(CASE WHEN complaint_rank_desc = 1 AND month_key_num IS NOT NULL THEN month_key_num END) AS complaint_min_month,
        MAX(CASE WHEN order_score_rank = 1 AND month_key_num IS NOT NULL THEN month_key_num END) AS order_max_month,
        MAX(CASE WHEN order_score_rank_desc = 1 AND month_key_num IS NOT NULL THEN month_key_num END) AS order_min_month,
        MAX(CASE WHEN satisfaction_rank = 1 AND month_key_num IS NOT NULL THEN month_key_num END) AS satisfaction_max_month,
        MAX(CASE WHEN satisfaction_rank_desc = 1 AND month_key_num IS NOT NULL THEN month_key_num END) AS satisfaction_min_month
    FROM (
        SELECT
            cms_code_enc,
            month_key_num,
            item_sum,
            revenue_sum,
            complaint_sum,
            order_score_avg,
            satisfaction_avg,
            ROW_NUMBER() OVER (PARTITION BY cms_code_enc ORDER BY item_sum DESC) AS item_count_rank,
            ROW_NUMBER() OVER (PARTITION BY cms_code_enc ORDER BY item_sum ASC) AS item_count_rank_desc,
            ROW_NUMBER() OVER (PARTITION BY cms_code_enc ORDER BY revenue_sum DESC) AS revenue_rank,
            ROW_NUMBER() OVER (PARTITION BY cms_code_enc ORDER BY revenue_sum ASC) AS revenue_rank_desc,
            ROW_NUMBER() OVER (PARTITION BY cms_code_enc ORDER BY COALESCE(complaint_sum, -1) DESC) AS complaint_rank,
            ROW_NUMBER() OVER (PARTITION BY cms_code_enc ORDER BY COALESCE(complaint_sum, -1) ASC) AS complaint_rank_desc,
            ROW_NUMBER() OVER (PARTITION BY cms_code_enc ORDER BY COALESCE(order_score_avg, -1) DESC) AS order_score_rank,
            ROW_NUMBER() OVER (PARTITION BY cms_code_enc ORDER BY COALESCE(order_score_avg, -1) ASC) AS order_score_rank_desc,
            ROW_NUMBER() OVER (PARTITION BY cms_code_enc ORDER BY COALESCE(satisfaction_avg, -1) DESC) AS satisfaction_rank,
            ROW_NUMBER() OVER (PARTITION BY cms_code_enc ORDER BY COALESCE(satisfaction_avg, -1) ASC) AS satisfaction_rank_desc
        FROM monthly_sums
    ) ranked
    GROUP BY cms_code_enc
),

base_agg AS (
    SELECT
        ms.cms_code_enc,
        COUNT(*) AS month_count,
        
        SUM(ms.item_sum)::bigint AS item_sum,
        AVG(ms.item_sum)::double precision AS item_avg,
        STDDEV_POP(ms.item_sum)::double precision AS item_std,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY ms.item_sum) AS item_median,
        MAX(ms.item_sum)::bigint AS item_max,
        MIN(ms.item_sum)::bigint AS item_min,
        
        SUM(ms.revenue_sum)::bigint AS revenue_sum,
        AVG(ms.revenue_sum)::double precision AS revenue_avg,
        STDDEV_POP(ms.revenue_sum)::double precision AS revenue_std,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY ms.revenue_sum) AS revenue_median,
        MAX(ms.revenue_sum)::bigint AS revenue_max,
        MIN(ms.revenue_sum)::bigint AS revenue_min,
        
        SUM(ms.complaint_sum)::bigint AS complaint_sum,
        AVG(ms.complaint_sum)::double precision AS complaint_avg,
        STDDEV_POP(ms.complaint_sum)::double precision AS complaint_std,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY ms.complaint_sum) AS complaint_median,
        SUM(ms.months_with_complaint)::int AS months_with_complaint_ct,
        
        SUM(mm.weight_kg)::double precision AS weight_sum,
        AVG(mm.weight_kg)::double precision AS weight_avg,
        STDDEV_POP(mm.weight_kg)::double precision AS weight_std,
        
        SUM(ms.delay_sum)::double precision AS delay_sum,
        SUM(CASE WHEN ms.delay_sum > 0 THEN 1 END)::int AS delay_count_ct,
        SUM(mm.delay_day)::double precision AS delay_day_sum,
        SUM(mm.refunded)::double precision AS refund_sum,
        SUM(mm.noaccepted)::double precision AS noaccepted_sum,
        SUM(mm.lost_order)::double precision AS lost_sum,
        SUM(ms.nodone_sum)::double precision AS nodone_sum,
        
        SUM(mm.intra_province)::double precision AS intra_prov_sum,
        SUM(mm.international)::double precision AS intl_sum,
        
        AVG(ms.order_score_avg)::double precision AS order_score_avg,
        STDDEV_POP(ms.order_score_avg)::double precision AS order_score_std,
        AVG(ms.satisfaction_avg)::double precision AS satisfaction_avg,
        STDDEV_POP(ms.satisfaction_avg)::double precision AS satisfaction_std,
        
        COUNT(DISTINCT CASE WHEN ms.item_sum > 0 THEN ms.month_key END)::int AS active_months,
        AVG(mm.lastday)::double precision AS avg_lastday,
        
        SUM(mm.ser_c)::bigint AS ser_c_sum,
        SUM(mm.ser_e)::bigint AS ser_e_sum,
        SUM(mm.ser_m)::bigint AS ser_m_sum,
        SUM(mm.ser_p)::bigint AS ser_p_sum,
        SUM(mm.ser_r)::bigint AS ser_r_sum,
        SUM(mm.ser_u)::bigint AS ser_u_sum,
        SUM(mm.ser_l)::bigint AS ser_l_sum,
        SUM(mm.ser_q)::bigint AS ser_q_sum
    FROM monthly_sums ms
    LEFT JOIN monthly_metrics mm ON mm.cms_code_enc = ms.cms_code_enc 
        AND to_char(mm.report_month, 'YYMM') = ms.month_key
    GROUP BY ms.cms_code_enc
),

-- Pre-compute service mix to avoid repeated GREATEST() calls
service_mix_stats AS (
    SELECT
        b.cms_code_enc,
        ((CASE WHEN b.ser_c_sum > 0 THEN 1 ELSE 0 END) +
         (CASE WHEN b.ser_e_sum > 0 THEN 1 ELSE 0 END) +
         (CASE WHEN b.ser_m_sum > 0 THEN 1 ELSE 0 END) +
         (CASE WHEN b.ser_p_sum > 0 THEN 1 ELSE 0 END) +
         (CASE WHEN b.ser_r_sum > 0 THEN 1 ELSE 0 END) +
         (CASE WHEN b.ser_u_sum > 0 THEN 1 ELSE 0 END) +
         (CASE WHEN b.ser_l_sum > 0 THEN 1 ELSE 0 END) +
         (CASE WHEN b.ser_q_sum > 0 THEN 1 ELSE 0 END))::int AS service_types_used,
        GREATEST(b.ser_c_sum, b.ser_e_sum, b.ser_m_sum, b.ser_p_sum, b.ser_r_sum, b.ser_u_sum, b.ser_l_sum, b.ser_q_sum)::bigint AS max_service_sum,
        CASE 
            WHEN b.ser_c_sum = GREATEST(b.ser_c_sum, b.ser_e_sum, b.ser_m_sum, b.ser_p_sum, b.ser_r_sum, b.ser_u_sum, b.ser_l_sum, b.ser_q_sum) THEN 'C'
            WHEN b.ser_e_sum = GREATEST(b.ser_c_sum, b.ser_e_sum, b.ser_m_sum, b.ser_p_sum, b.ser_r_sum, b.ser_u_sum, b.ser_l_sum, b.ser_q_sum) THEN 'E'
            WHEN b.ser_m_sum = GREATEST(b.ser_c_sum, b.ser_e_sum, b.ser_m_sum, b.ser_p_sum, b.ser_r_sum, b.ser_u_sum, b.ser_l_sum, b.ser_q_sum) THEN 'M'
            WHEN b.ser_p_sum = GREATEST(b.ser_c_sum, b.ser_e_sum, b.ser_m_sum, b.ser_p_sum, b.ser_r_sum, b.ser_u_sum, b.ser_l_sum, b.ser_q_sum) THEN 'P'
            WHEN b.ser_r_sum = GREATEST(b.ser_c_sum, b.ser_e_sum, b.ser_m_sum, b.ser_p_sum, b.ser_r_sum, b.ser_u_sum, b.ser_l_sum, b.ser_q_sum) THEN 'R'
            WHEN b.ser_u_sum = GREATEST(b.ser_c_sum, b.ser_e_sum, b.ser_m_sum, b.ser_p_sum, b.ser_r_sum, b.ser_u_sum, b.ser_l_sum, b.ser_q_sum) THEN 'U'
            WHEN b.ser_l_sum = GREATEST(b.ser_c_sum, b.ser_e_sum, b.ser_m_sum, b.ser_p_sum, b.ser_r_sum, b.ser_u_sum, b.ser_l_sum, b.ser_q_sum) THEN 'L'
            WHEN b.ser_q_sum = GREATEST(b.ser_c_sum, b.ser_e_sum, b.ser_m_sum, b.ser_p_sum, b.ser_r_sum, b.ser_u_sum, b.ser_l_sum, b.ser_q_sum) THEN 'Q'
            ELSE 'E'
        END::varchar AS dominant_service
    FROM base_agg b
),

complaint_stats AS (
    SELECT
        cms_code_enc,
        COUNT(DISTINCT complaint_code)::int AS complaint_diversity,
        (array_agg(complaint_code ORDER BY cnt DESC))[1]::int AS most_common_complaint
    FROM (
        SELECT
            cms_code_enc,
            complaint_code,
            COUNT(*) AS cnt
        FROM {COMPLAINT_SOURCE_TABLE}
        WHERE create_complaint_date >= DATE '{START_DATE}' AND create_complaint_date <= DATE '{END_DATE}'
        GROUP BY cms_code_enc, complaint_code
    ) complaint_counts
    GROUP BY cms_code_enc
),

bccp_stats AS (
    SELECT
        cms_code_enc,
        COUNT(DISTINCT DATE(sending_time))::int AS active_days,
        (DATE '{END_DATE}'::date - DATE '{START_DATE}'::date + 1)::int AS window_days,
        (DATE '{END_DATE}'::date - MAX(DATE(sending_time)))::int AS recency_days
    FROM {BCCP_SOURCE_TABLE}
    WHERE sending_time >= DATE '{START_DATE}' AND sending_time <= DATE '{END_DATE}'
    GROUP BY cms_code_enc
),

slopes AS (
    SELECT
        cms_code_enc,
        COALESCE(regr_slope(item_sum, month_key_num), 0)::double precision AS item_slope,
        COALESCE(regr_slope(revenue_sum, month_key_num), 0)::double precision AS revenue_slope,
        COALESCE(regr_slope(satisfaction_avg, month_key_num), 0)::double precision AS satisfy_slope,
        COALESCE(regr_slope(complaint_sum, month_key_num), 0)::double precision AS complaint_slope
    FROM monthly_sums
    GROUP BY cms_code_enc
)

INSERT INTO {TABLE_NAME} (
    cms_code_enc, window_size, window_start, window_end,
    {MONTHLY_COLUMNS_LIST},
    item_sum, item_avg, item_std, item_min_month, item_max_month, item_median,
    revenue_sum, revenue_avg, revenue_std, revenue_min_month, revenue_max_month, revenue_median,
    complaint_sum, complaint_avg, complaint_std, complaint_min_month, complaint_max_month, complaint_median, complaint_diversity, most_common_complaint,
    weight_sum, weight_avg, weight_std,
    avg_revenue_per_item, avg_weight_per_item,
    pct_delay, pct_refund, pct_noaccepted, pct_lost_order, pct_complaint, pct_complaint_per_item, pct_successful_item, pct_intra_province, pct_international, avg_delayday,
    order_score_avg, order_score_std, order_min_month, order_max_month,
    satisfaction_avg, satisfaction_std, satisfaction_min_month, satisfaction_max_month,
    active_months, inactive_months,
    active_days, inactive_days, avg_noservice_days, max_consecutive_inactive, avg_lastday,
    ser_c_sum, ser_e_sum, ser_m_sum, ser_p_sum, ser_r_sum, ser_u_sum, ser_l_sum, ser_q_sum,
    item_slope, revenue_slope, satisfy_slope, complaint_slope,
    cv_item, cv_revenue, item_range, revenue_range,
    service_types_used, dominant_service, dominant_service_ratio, 
    recency, frequency, monetary
)
SELECT
    b.cms_code_enc,
    {WINDOW_SIZE}::int,
    '{START_YM}'::varchar,
    '{END_YM}'::varchar,

    {MONTHLY_SELECT_COLUMNS},

    b.item_sum, b.item_avg, b.item_std, m.item_min_month,m.item_max_month, b.item_median,

    b.revenue_sum, b.revenue_avg, b.revenue_std, m.revenue_min_month, m.revenue_max_month, b.revenue_median,

    b.complaint_sum, b.complaint_avg, b.complaint_std, m.complaint_min_month, m.complaint_max_month, b.complaint_median,
    COALESCE(cs.complaint_diversity, 0), cs.most_common_complaint,

    b.weight_sum, b.weight_avg, b.weight_std,

    COALESCE(b.revenue_sum::double precision / NULLIF(b.item_sum, 0), 0),
    COALESCE(b.weight_sum::double precision / NULLIF(b.item_sum, 0), 0),

    COALESCE(b.delay_sum::double precision / NULLIF(b.item_sum, 0), 0),
    COALESCE(b.refund_sum::double precision / NULLIF(b.item_sum, 0), 0),
    COALESCE(b.noaccepted_sum::double precision / NULLIF(b.item_sum, 0), 0),
    COALESCE(b.lost_sum::double precision / NULLIF(b.item_sum, 0), 0),
    COALESCE(b.complaint_sum::double precision / NULLIF(b.item_sum, 0), 0),
    COALESCE(b.months_with_complaint_ct::double precision / NULLIF(b.month_count, 0), 0),
    COALESCE((b.item_sum::double precision - b.nodone_sum) / NULLIF(b.item_sum, 0), 0),
    COALESCE(b.intra_prov_sum::double precision / NULLIF(b.item_sum, 0), 0),
    COALESCE(b.intl_sum::double precision / NULLIF(b.item_sum, 0), 0),
    CASE WHEN b.delay_count_ct > 0 THEN (b.delay_day_sum::double precision / b.delay_count_ct) ELSE 0 END,

    b.order_score_avg, b.order_score_std, m.order_min_month, m.order_max_month,
    b.satisfaction_avg, b.satisfaction_std, m.satisfaction_min_month, m.satisfaction_max_month,

    b.active_months,
    ({WINDOW_SIZE} - b.active_months)::int,
    COALESCE(bs.active_days, 0),
    (bs.window_days::int - COALESCE(bs.active_days, 0))::int,
    ((bs.window_days::int - COALESCE(bs.active_days, 0))::double precision / GREATEST(CEIL(bs.window_days::int / 30.0), 1))::double precision,
    0::int,
    b.avg_lastday,

    b.ser_c_sum, b.ser_e_sum, b.ser_m_sum, b.ser_p_sum, b.ser_r_sum, b.ser_u_sum, b.ser_l_sum, b.ser_q_sum,

    COALESCE(sl.item_slope, 0), COALESCE(sl.revenue_slope, 0), COALESCE(sl.satisfy_slope, 0), COALESCE(sl.complaint_slope, 0),

    CASE WHEN b.item_avg <> 0 THEN COALESCE(b.item_std / b.item_avg, 0) ELSE 0 END,
    CASE WHEN b.revenue_avg <> 0 THEN COALESCE(b.revenue_std / b.revenue_avg, 0) ELSE 0 END,
    (b.item_max - b.item_min)::bigint,
    (b.revenue_max - b.revenue_min)::bigint,

    sms.service_types_used,
    sms.dominant_service,
    CASE WHEN (b.ser_c_sum + b.ser_e_sum + b.ser_m_sum + b.ser_p_sum + b.ser_r_sum + b.ser_u_sum + b.ser_l_sum + b.ser_q_sum) > 0 
         THEN sms.max_service_sum::double precision / 
              (b.ser_c_sum + b.ser_e_sum + b.ser_m_sum + b.ser_p_sum + b.ser_r_sum + b.ser_u_sum + b.ser_l_sum + b.ser_q_sum)
         ELSE 0 END::double precision,

    COALESCE(bs.recency_days, 0)::int,
    CASE WHEN b.active_months > 0 THEN (b.item_sum::double precision / b.active_months) ELSE 0 END,
    CASE WHEN b.active_months > 0 THEN (b.revenue_sum::double precision / b.active_months) ELSE 0 END

FROM base_agg b
LEFT JOIN monthly_pivoted mp ON mp.cms_code_enc = b.cms_code_enc
LEFT JOIN month_extremes m ON m.cms_code_enc = b.cms_code_enc
LEFT JOIN complaint_stats cs ON cs.cms_code_enc = b.cms_code_enc
LEFT JOIN bccp_stats bs ON bs.cms_code_enc = b.cms_code_enc
LEFT JOIN slopes sl ON sl.cms_code_enc = b.cms_code_enc
LEFT JOIN service_mix_stats sms ON sms.cms_code_enc = b.cms_code_enc;

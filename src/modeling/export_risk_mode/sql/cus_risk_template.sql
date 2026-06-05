-- Template for cus_risk table creation (Simplified)
-- Replace {THRESHOLD} with actual threshold value (e.g., 70)
-- Replace {TABLE_NAME} with table name (e.g., cus_risk_70)  

CREATE SCHEMA IF NOT EXISTS data_static;

-- Snapshot & History table: stores risk scores
CREATE TABLE IF NOT EXISTS data_static.{TABLE_NAME} (
    cms_code_enc              TEXT NOT NULL,
    predict_period            INT NOT NULL,
    
    window_end                INT,
    
    -- Window features (_last refers to most recent month in data)
    item_last                 DOUBLE PRECISION,
    revenue_last              DOUBLE PRECISION,
    complaint_last            DOUBLE PRECISION,
    delay_last                DOUBLE PRECISION,
    nodone_last               DOUBLE PRECISION,
    order_score_last          DOUBLE PRECISION,
    satisfaction_last         DOUBLE PRECISION,

    -- Risk scores:
    -- churn_rate is a CRM-facing display risk score percentile in 0-100.
    -- model_probability_pct is the raw model probability percent for audit.
    churn_rate                DOUBLE PRECISION NOT NULL,
    model_probability_pct     DOUBLE PRECISION,

    -- Simple reasons (top 3, prioritized)
    reason_1                  TEXT,
    reason_2                  TEXT,
    reason_3                  TEXT,

    update_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    PRIMARY KEY (cms_code_enc, predict_period)
);

-- Index for ordering (newest period first, then high risk first)
CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_predict_risk ON data_static.{TABLE_NAME} (predict_period DESC, churn_rate DESC);

ALTER TABLE data_static.{TABLE_NAME}
ADD COLUMN IF NOT EXISTS model_probability_pct DOUBLE PRECISION;

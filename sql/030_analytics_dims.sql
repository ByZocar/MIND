-- =====================================================================
-- 030_analytics_dims.sql
-- Dimensiones del modelo estrella.
-- =====================================================================

-- ---------- dim_patient ----------
DROP TABLE IF EXISTS analytics.dim_patient CASCADE;

CREATE TABLE analytics.dim_patient (
    patient_sk      BIGSERIAL    PRIMARY KEY,
    patient_id      TEXT         NOT NULL,
    dataset_origin  TEXT         NOT NULL CHECK (dataset_origin IN ('train','test')),
    sex             CHAR(1)      NOT NULL CHECK (sex IN ('M','F')),
    age_years       INTEGER      NOT NULL,
    age_bucket      TEXT         NOT NULL,
    first_loaded_at TIMESTAMPTZ  NOT NULL,
    UNIQUE (patient_id, dataset_origin)
);

CREATE INDEX IF NOT EXISTS ix_dim_patient_origin ON analytics.dim_patient (dataset_origin);
CREATE INDEX IF NOT EXISTS ix_dim_patient_sex    ON analytics.dim_patient (sex);

COMMENT ON TABLE analytics.dim_patient IS
    'Una fila por paciente. La unicidad es (patient_id, dataset_origin) porque train↔test no comparten pacientes pero el modelo es agnóstico al origen.';
COMMENT ON COLUMN analytics.dim_patient.age_bucket IS
    'Bandas etarias clínicamente útiles: <50, 50-64, 65-79, 80+.';


-- ---------- dim_severity ----------
DROP TABLE IF EXISTS analytics.dim_severity CASCADE;

CREATE TABLE analytics.dim_severity (
    severity_sk   SERIAL  PRIMARY KEY,
    nihss_band    TEXT    NOT NULL,
    aspects_band  TEXT    NOT NULL,
    UNIQUE (nihss_band, aspects_band)
);

COMMENT ON TABLE analytics.dim_severity IS
    'Bandas clínicas para NIHSS y ASPECTS. NIHSS: 0=none, 1-4=minor, 5-15=moderate, 16-20=mod-severe, 21+=severe. ASPECTS: <=7 = significant ischemia, 8-10 = preserved.';

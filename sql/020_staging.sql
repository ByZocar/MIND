-- =====================================================================
-- 020_staging.sql
-- Tabla staging.fct_slice_stg: tipado fuerte, renombres documentados,
-- columnas derivadas (slice_order, evolution_hours, missing indicators).
--
-- Patrón: TRUNCATE + INSERT desde raw (último load_id por dataset_origin).
-- Esto se invoca desde el loader (Python) o desde el DAG (Airflow, Fase 3).
-- =====================================================================

DROP TABLE IF EXISTS staging.fct_slice_stg CASCADE;

CREATE TABLE staging.fct_slice_stg (
    slice_sk                BIGSERIAL    PRIMARY KEY,
    load_id                 BIGINT       NOT NULL,
    dataset_origin          TEXT         NOT NULL,
    row_order               INTEGER      NOT NULL,
    patient_id              TEXT         NOT NULL,
    slice_order             INTEGER      NOT NULL,
    -- shape2D
    shape2d_meshsurface             DOUBLE PRECISION,
    shape2d_perimeter               DOUBLE PRECISION,
    shape2d_sphericity              DOUBLE PRECISION,
    shape2d_sphericaldisproportion  DOUBLE PRECISION,
    shape2d_maximumdiameter         DOUBLE PRECISION,
    shape2d_majoraxislength         DOUBLE PRECISION,
    shape2d_minoraxislength         DOUBLE PRECISION,
    shape2d_elongation              DOUBLE PRECISION,
    -- firstorder
    firstorder_entropy                       DOUBLE PRECISION,
    firstorder_minimum                       DOUBLE PRECISION,
    firstorder_10percentile                  DOUBLE PRECISION,
    firstorder_90percentile                  DOUBLE PRECISION,
    firstorder_maximum                       DOUBLE PRECISION,
    firstorder_mean                          DOUBLE PRECISION,
    firstorder_median                        DOUBLE PRECISION,
    firstorder_interquartilerange            DOUBLE PRECISION,
    firstorder_range                         DOUBLE PRECISION,
    firstorder_meanabsolutedeviation         DOUBLE PRECISION,
    firstorder_robustmeanabsolutedeviation   DOUBLE PRECISION,
    firstorder_rootmeansquared               DOUBLE PRECISION,
    firstorder_standarddeviation             DOUBLE PRECISION,
    firstorder_skewness                      DOUBLE PRECISION,
    firstorder_kurtosis                      DOUBLE PRECISION,
    firstorder_variance                      DOUBLE PRECISION,
    firstorder_uniformity                    DOUBLE PRECISION,
    -- glcm
    glcm_autocorrelation                     DOUBLE PRECISION,
    glcm_jointaverage                        DOUBLE PRECISION,
    glcm_clusterprominence                   DOUBLE PRECISION,
    glcm_clustershade                        DOUBLE PRECISION,
    glcm_clustertendency                     DOUBLE PRECISION,
    glcm_contrast                            DOUBLE PRECISION,
    glcm_correlation                         DOUBLE PRECISION,
    glcm_differenceaverage                   DOUBLE PRECISION,
    glcm_differenceentropy                   DOUBLE PRECISION,
    glcm_differencevariance                  DOUBLE PRECISION,
    glcm_jointenergy                         DOUBLE PRECISION,
    glcm_jointentropy                        DOUBLE PRECISION,
    glcm_imc1                                DOUBLE PRECISION,
    glcm_imc2                                DOUBLE PRECISION,
    glcm_idm                                 DOUBLE PRECISION,
    glcm_mcc                                 DOUBLE PRECISION,
    glcm_idmn                                DOUBLE PRECISION,
    glcm_id                                  DOUBLE PRECISION,
    glcm_idn                                 DOUBLE PRECISION,
    glcm_maximumprobability                  DOUBLE PRECISION,
    glcm_sumaverage                          DOUBLE PRECISION,
    glcm_sumentropy                          DOUBLE PRECISION,
    glcm_sumsquares                          DOUBLE PRECISION,
    -- glszm
    glszm_smallareaemphasis                  DOUBLE PRECISION,
    glszm_largeareaemphasis                  DOUBLE PRECISION,
    glszm_graylevelnonuniformity             DOUBLE PRECISION,
    glszm_graylevelnonuniformitynormalized   DOUBLE PRECISION,
    glszm_sizezonenonuniformity              DOUBLE PRECISION,
    glszm_sizezonenonuniformitynormalized    DOUBLE PRECISION,
    glszm_zonepercentage                     DOUBLE PRECISION,
    glszm_graylevelvariance                  DOUBLE PRECISION,
    glszm_zonevariance                       DOUBLE PRECISION,
    glszm_zoneentropy                        DOUBLE PRECISION,
    glszm_lowgraylevelzoneemphasis           DOUBLE PRECISION,
    glszm_highgraylevelzoneemphasis          DOUBLE PRECISION,
    glszm_smallarealowgraylevelemphasis      DOUBLE PRECISION,
    glszm_smallareahighgraylevelemphasis     DOUBLE PRECISION,
    glszm_largearealowgraylevelemphasis      DOUBLE PRECISION,
    glszm_largeareahighgraylevelemphasis     DOUBLE PRECISION,
    -- demo + clinical
    age_years               INTEGER,
    sex                     CHAR(1)      CHECK (sex IN ('M','F')),
    nihss                   DOUBLE PRECISION,
    nihss_was_missing       BOOLEAN      NOT NULL,
    aspects                 INTEGER,
    aspects_was_missing     BOOLEAN      NOT NULL,
    evolution_minutes       DOUBLE PRECISION,
    evolution_hours         DOUBLE PRECISION,
    is_over_window          SMALLINT     NOT NULL CHECK (is_over_window IN (0,1)),
    loaded_at               TIMESTAMPTZ  NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_stg_patient_id     ON staging.fct_slice_stg (patient_id);
CREATE INDEX IF NOT EXISTS ix_stg_dataset_origin ON staging.fct_slice_stg (dataset_origin);
CREATE INDEX IF NOT EXISTS ix_stg_target         ON staging.fct_slice_stg (is_over_window);

COMMENT ON TABLE staging.fct_slice_stg IS
    'Grano: 1 slice. Origen: raw.acv_slice_csv (último load_id). evolution_minutes/hours conviven; ambas son leakage del target — NUNCA como feature.';
COMMENT ON COLUMN staging.fct_slice_stg.slice_order IS
    'row_number() OVER (PARTITION BY (dataset_origin, patient_id) ORDER BY row_order) — sintetiza orden de slice ya que la fuente no lo trae.';
COMMENT ON COLUMN staging.fct_slice_stg.evolution_minutes IS
    'Minutos desde inicio de síntomas (unidad confirmada por EDA Fase 1, ADR-009).';
COMMENT ON COLUMN staging.fct_slice_stg.evolution_hours IS
    'Derivada: evolution_minutes / 60.0. Para reporte clínico únicamente.';

-- Tabla de cuarentena: filas que no pueden tipar correctamente quedan aquí
-- (en Fase 3 las llenará Great Expectations al fallar una expectation).
CREATE TABLE IF NOT EXISTS staging.quarantine_slice (
    quarantine_sk   BIGSERIAL    PRIMARY KEY,
    load_id         BIGINT       NOT NULL,
    row_order       INTEGER      NOT NULL,
    dataset_origin  TEXT         NOT NULL,
    raw_payload     JSONB        NOT NULL,
    reason          TEXT         NOT NULL,
    quarantined_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

COMMENT ON TABLE staging.quarantine_slice IS
    'Filas rechazadas por reglas de calidad. Se mantienen para auditoría; no llegan a analytics.';

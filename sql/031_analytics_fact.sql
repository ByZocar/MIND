-- =====================================================================
-- 031_analytics_fact.sql
-- Tabla de hechos a nivel slice + vista materializada a nivel paciente.
-- =====================================================================

DROP MATERIALIZED VIEW IF EXISTS analytics.agg_patient CASCADE;
DROP TABLE              IF EXISTS analytics.fct_slice CASCADE;

CREATE TABLE analytics.fct_slice (
    slice_sk            BIGSERIAL    PRIMARY KEY,
    patient_sk          BIGINT       NOT NULL REFERENCES analytics.dim_patient(patient_sk),
    severity_sk         INTEGER      REFERENCES analytics.dim_severity(severity_sk),
    dataset_origin      TEXT         NOT NULL,
    slice_order         INTEGER      NOT NULL,
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
    -- clinical (sin features-leakage)
    nihss                   DOUBLE PRECISION,
    nihss_was_missing       BOOLEAN          NOT NULL,
    aspects                 INTEGER,
    aspects_was_missing     BOOLEAN          NOT NULL,
    -- columnas etiquetadas como leakage (sólo reporte clínico, NO features)
    evolution_minutes       DOUBLE PRECISION,
    evolution_hours         DOUBLE PRECISION,
    is_over_window          SMALLINT         NOT NULL CHECK (is_over_window IN (0,1)),
    loaded_at               TIMESTAMPTZ      NOT NULL,
    UNIQUE (patient_sk, slice_order)
);

CREATE INDEX IF NOT EXISTS ix_fct_patient_sk     ON analytics.fct_slice (patient_sk);
CREATE INDEX IF NOT EXISTS ix_fct_dataset_origin ON analytics.fct_slice (dataset_origin);
CREATE INDEX IF NOT EXISTS ix_fct_target         ON analytics.fct_slice (is_over_window);

COMMENT ON TABLE  analytics.fct_slice IS 'Tabla de hechos. Grano: 1 slice de TC.';
COMMENT ON COLUMN analytics.fct_slice.evolution_minutes IS 'LEAKAGE — no usar como feature.';
COMMENT ON COLUMN analytics.fct_slice.evolution_hours   IS 'LEAKAGE — no usar como feature.';


-- ---------- agg_patient ----------
-- Vista materializada a nivel paciente.
-- En Fase 2 incluimos agregados clínicamente útiles (no exhaustivos);
-- la Fase 5 construirá la matriz completa de features ML en analytics.features_v1.

CREATE MATERIALIZED VIEW analytics.agg_patient AS
SELECT
    p.patient_sk,
    p.patient_id,
    p.dataset_origin,
    p.sex,
    p.age_years,
    p.age_bucket,
    -- target (es constante por paciente; ver expectativa de integridad)
    MAX(f.is_over_window)::SMALLINT  AS is_over_window,
    COUNT(*)                         AS n_slices,
    -- clínicas (valor constante por paciente en esta fuente)
    MAX(f.nihss)                     AS nihss,
    BOOL_OR(f.nihss_was_missing)     AS nihss_was_missing,
    MAX(f.aspects)                   AS aspects,
    BOOL_OR(f.aspects_was_missing)   AS aspects_was_missing,
    MAX(f.evolution_minutes)         AS evolution_minutes,
    MAX(f.evolution_hours)           AS evolution_hours,
    -- proxy de lesión: máximo área 2D entre slices del paciente
    MAX(f.shape2d_meshsurface)       AS lesion_max_area,
    AVG(f.shape2d_meshsurface)       AS lesion_avg_area,
    AVG(f.shape2d_maximumdiameter)   AS lesion_avg_maxdiameter,
    -- heterogeneidad textura
    AVG(f.firstorder_entropy)        AS firstorder_entropy_avg,
    STDDEV_SAMP(f.firstorder_entropy) AS firstorder_entropy_std,
    AVG(f.firstorder_mean)           AS firstorder_intensity_avg,
    AVG(f.glcm_contrast)             AS glcm_contrast_avg,
    AVG(f.glszm_zoneentropy)         AS glszm_zoneentropy_avg
FROM analytics.fct_slice    f
JOIN analytics.dim_patient  p ON p.patient_sk = f.patient_sk
GROUP BY p.patient_sk, p.patient_id, p.dataset_origin, p.sex, p.age_years, p.age_bucket;

CREATE UNIQUE INDEX IF NOT EXISTS ux_agg_patient_sk ON analytics.agg_patient (patient_sk);
CREATE INDEX        IF NOT EXISTS ix_agg_origin    ON analytics.agg_patient (dataset_origin);
CREATE INDEX        IF NOT EXISTS ix_agg_target    ON analytics.agg_patient (is_over_window);

COMMENT ON MATERIALIZED VIEW analytics.agg_patient IS
    'Una fila por paciente. Refrescar con REFRESH MATERIALIZED VIEW CONCURRENTLY tras cargar el fact. Fase 5 expandirá los agregados.';

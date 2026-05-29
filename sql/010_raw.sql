-- =====================================================================
-- 010_raw.sql
-- Tabla raw.acv_slice_csv: ingesta inmutable, append-only por load_id.
-- Todas las columnas son TEXT para máxima tolerancia al CSV original.
-- El tipado real ocurre en staging.
-- =====================================================================

CREATE TABLE IF NOT EXISTS raw.acv_slice_csv (
    load_id           BIGINT       NOT NULL,
    loaded_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    dataset_origin    TEXT         NOT NULL CHECK (dataset_origin IN ('train','test')),
    row_order         INTEGER      NOT NULL,
    "ID"                                           TEXT,
    "original_shape2D_MeshSurface"                 TEXT,
    "original_shape2D_Perimeter"                   TEXT,
    "original_shape2D_Sphericity"                  TEXT,
    "original_shape2D_SphericalDisproportion"      TEXT,
    "original_shape2D_MaximumDiameter"             TEXT,
    "original_shape2D_MajorAxisLength"             TEXT,
    "original_shape2D_MinorAxisLength"             TEXT,
    "original_shape2D_Elongation"                  TEXT,
    "original_firstorder_Entropy"                  TEXT,
    "original_firstorder_Minimum"                  TEXT,
    "original_firstorder_10Percentile"             TEXT,
    "original_firstorder_90Percentile"             TEXT,
    "original_firstorder_Maximum"                  TEXT,
    "original_firstorder_Mean"                     TEXT,
    "original_firstorder_Median"                   TEXT,
    "original_firstorder_InterquartileRange"       TEXT,
    "original_firstorder_Range"                    TEXT,
    "original_firstorder_MeanAbsoluteDeviation"    TEXT,
    "original_firstorder_RobustMeanAbsoluteDeviation" TEXT,
    "original_firstorder_RootMeanSquared"          TEXT,
    "original_firstorder_StandardDeviation"        TEXT,
    "original_firstorder_Skewness"                 TEXT,
    "original_firstorder_Kurtosis"                 TEXT,
    "original_firstorder_Variance"                 TEXT,
    "original_firstorder_Uniformity"               TEXT,
    "original_glcm_Autocorrelation"                TEXT,
    "original_glcm_JointAverage"                   TEXT,
    "original_glcm_ClusterProminence"              TEXT,
    "original_glcm_ClusterShade"                   TEXT,
    "original_glcm_ClusterTendency"                TEXT,
    "original_glcm_Contrast"                       TEXT,
    "original_glcm_Correlation"                    TEXT,
    "original_glcm_DifferenceAverage"              TEXT,
    "original_glcm_DifferenceEntropy"              TEXT,
    "original_glcm_DifferenceVariance"             TEXT,
    "original_glcm_JointEnergy"                    TEXT,
    "original_glcm_JointEntropy"                   TEXT,
    "original_glcm_Imc1"                           TEXT,
    "original_glcm_Imc2"                           TEXT,
    "original_glcm_Idm"                            TEXT,
    "original_glcm_MCC"                            TEXT,
    "original_glcm_Idmn"                           TEXT,
    "original_glcm_Id"                             TEXT,
    "original_glcm_Idn"                            TEXT,
    "original_glcm_MaximumProbability"             TEXT,
    "original_glcm_SumAverage"                     TEXT,
    "original_glcm_SumEntropy"                     TEXT,
    "original_glcm_SumSquares"                     TEXT,
    "original_glszm_SmallAreaEmphasis"             TEXT,
    "original_glszm_LargeAreaEmphasis"             TEXT,
    "original_glszm_GrayLevelNonUniformity"        TEXT,
    "original_glszm_GrayLevelNonUniformityNormalized" TEXT,
    "original_glszm_SizeZoneNonUniformity"         TEXT,
    "original_glszm_SizeZoneNonUniformityNormalized" TEXT,
    "original_glszm_ZonePercentage"                TEXT,
    "original_glszm_GrayLevelVariance"             TEXT,
    "original_glszm_ZoneVariance"                  TEXT,
    "original_glszm_ZoneEntropy"                   TEXT,
    "original_glszm_LowGrayLevelZoneEmphasis"      TEXT,
    "original_glszm_HighGrayLevelZoneEmphasis"     TEXT,
    "original_glszm_SmallAreaLowGrayLevelEmphasis" TEXT,
    "original_glszm_SmallAreaHighGrayLevelEmphasis" TEXT,
    "original_glszm_LargeAreaLowGrayLevelEmphasis" TEXT,
    "original_glszm_LargeAreaHighGrayLevelEmphasis" TEXT,
    "Patien Age in Study"   TEXT,
    "Patient Sex"           TEXT,
    "NIHSS"                 TEXT,
    "Evolution Time"        TEXT,
    "ASPECTS"               TEXT,
    "Evolution Time_Clas"   TEXT
);

CREATE INDEX IF NOT EXISTS ix_raw_load_id        ON raw.acv_slice_csv (load_id);
CREATE INDEX IF NOT EXISTS ix_raw_dataset_origin ON raw.acv_slice_csv (dataset_origin);
CREATE INDEX IF NOT EXISTS ix_raw_id             ON raw.acv_slice_csv ("ID");

COMMENT ON TABLE  raw.acv_slice_csv IS 'Inmutable. Cada carga del CSV agrega filas con un load_id único.';
COMMENT ON COLUMN raw.acv_slice_csv.row_order   IS 'Posición original de la fila en el archivo CSV (1-based).';
COMMENT ON COLUMN raw.acv_slice_csv.load_id     IS 'Identificador del batch de carga; permite rollback puntual.';

-- Catálogo de cargas (útil para Airflow y para auditoría)
CREATE TABLE IF NOT EXISTS raw.load_catalog (
    load_id        BIGINT       PRIMARY KEY,
    started_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ,
    source         TEXT         NOT NULL,
    rows_inserted  INTEGER,
    status         TEXT         NOT NULL DEFAULT 'running' CHECK (status IN ('running','ok','failed')),
    notes          TEXT
);

COMMENT ON TABLE raw.load_catalog IS 'Una fila por ejecución del loader. Cierra el ciclo de auditoría.';

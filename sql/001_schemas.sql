-- =====================================================================
-- 001_schemas.sql
-- Crea los tres schemas del Data Warehouse:
--   raw       -> ingesta inmutable (TEXT-friendly, append-only)
--   staging   -> tipado, limpieza, cuarentena
--   analytics -> modelo dimensional (estrella) listo para consumo
-- =====================================================================

CREATE SCHEMA IF NOT EXISTS raw       AUTHORIZATION acv_admin;
CREATE SCHEMA IF NOT EXISTS staging   AUTHORIZATION acv_admin;
CREATE SCHEMA IF NOT EXISTS analytics AUTHORIZATION acv_admin;

COMMENT ON SCHEMA raw       IS 'Ingesta cruda inmutable. No transformaciones. Append-only por load_id.';
COMMENT ON SCHEMA staging   IS 'Limpieza, tipado, deduplicación. Puede tener tablas de cuarentena.';
COMMENT ON SCHEMA analytics IS 'Modelo dimensional en estrella. Capa de consumo (BI, ML, KPIs).';

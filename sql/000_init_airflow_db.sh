#!/bin/bash
# =====================================================================
# 000_init_airflow_db.sh
# Crea la base de datos 'airflow' usada por el metastore de Apache Airflow.
# Corre como parte del docker-entrypoint-initdb.d en la primera inicialización
# del volumen pg_data. Para re-aplicarlo en un volumen ya inicializado, ver
# `make airflow-prepare` (idempotente) o ejecutar manualmente:
#
#   docker exec -u postgres acv-postgres psql -c "CREATE DATABASE airflow OWNER acv_admin"
# =====================================================================
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE airflow OWNER ' || quote_ident('$POSTGRES_USER')
    WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'airflow')\gexec
EOSQL

echo "  airflow database ensured."

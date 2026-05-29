"""Project-wide settings driven by environment variables.

This module is the single point of configuration for paths, database
credentials, Kafka, MLflow, and modeling defaults. Anything that varies
between environments lives here (and in `.env`), nowhere else.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor the .env path to the project root computed from THIS file's location,
# not the caller's CWD. Otherwise running a notebook from notebooks/ silently
# falls back to defaults (we hit this once — see ADR-016).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    project_root: Path = _PROJECT_ROOT

    # --- Database ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "acv"
    postgres_user: str = "acv_admin"
    postgres_password: str = "change_me_in_local"

    # --- Kafka ---
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_metrics_topic: str = "acv.metrics.v1"

    # --- MLflow ---
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "acv-window-classification"

    # --- Paths ---
    data_raw: Path = Field(default=Path("data/raw"))
    data_interim: Path = Field(default=Path("data/interim"))
    data_processed: Path = Field(default=Path("data/processed"))
    reports_dir: Path = Field(default=Path("reports"))

    # --- Modeling ---
    random_seed: int = 42
    target_recall_class1: float = 0.85
    min_specificity: float = 0.70

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()

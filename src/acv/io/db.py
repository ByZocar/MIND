"""Database engine factory.

Single source of truth for SQLAlchemy connections. The URL is built
from settings in `acv.config`. All ETL code uses `get_engine()`; never
build psycopg2 strings ad-hoc anywhere else.
"""
from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from acv.config import settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine for the project database."""
    return create_engine(
        settings.sqlalchemy_url,
        pool_pre_ping=True,
        future=True,
    )


def ping() -> str:
    """Light connectivity check. Returns the server version string."""
    from sqlalchemy import text

    with get_engine().connect() as conn:
        version = conn.execute(text("SELECT version()")).scalar_one()
    return str(version)

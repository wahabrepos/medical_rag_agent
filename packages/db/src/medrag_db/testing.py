"""Helpers for integration tests that need a real PostgreSQL + pgvector database.

Tests marked `integration` read TEST_DATABASE_URL (any database on the target
server) and run against a throwaway database created next to it, so a
development database is never touched.
"""

import os
from pathlib import Path

import psycopg
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url

ALEMBIC_INI = Path(__file__).resolve().parents[4] / "db" / "alembic.ini"


def get_test_database_url() -> str | None:
    return os.environ.get("TEST_DATABASE_URL")


def recreate_database(base_url: str, name: str) -> str:
    """Drop and create database `name` on base_url's server; return its URL."""
    url = make_url(base_url)
    conninfo = url.set(drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(conninfo, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{name}"')
    return url.set(database=name).render_as_string(hide_password=False)


def drop_database(base_url: str, name: str) -> None:
    url = make_url(base_url)
    conninfo = url.set(drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(conninfo, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def alembic_config(url: str) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


def migrate(url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str, revision: str = "base") -> None:
    command.downgrade(alembic_config(url), revision)

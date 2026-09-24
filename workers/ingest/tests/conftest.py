from collections.abc import Iterator

import pytest

from medrag_db.testing import drop_database, get_test_database_url, migrate, recreate_database

TEST_DB = "medrag_test_ingest"


@pytest.fixture(scope="module")
def database_url() -> Iterator[str]:
    """A freshly migrated throwaway database (integration tests only)."""
    base = get_test_database_url()
    if base is None:
        pytest.skip("TEST_DATABASE_URL is not set")
    url = recreate_database(base, TEST_DB)
    migrate(url)
    yield url
    drop_database(base, TEST_DB)

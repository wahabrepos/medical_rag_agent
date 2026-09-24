"""Vector index DDL, shared by the migration and the ingestion job.

One HNSW index per chunking profile (partial index), so each profile is searched
with its own graph. Ingestion drops the index before a bulk load and rebuilds it
afterwards, which is much faster than inserting into a live HNSW graph.
"""

PROFILES = ("parity", "standard")
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
HNSW_EF_SEARCH = 100


def hnsw_index_name(profile: str) -> str:
    if profile not in PROFILES:
        raise ValueError(f"unknown chunk profile: {profile}")
    return f"ix_chunks_embedding_hnsw_{profile}"


def create_hnsw_index_sql(profile: str) -> str:
    return (
        f"CREATE INDEX IF NOT EXISTS {hnsw_index_name(profile)} ON chunks "
        f"USING hnsw (embedding halfvec_cosine_ops) "
        f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION}) "
        f"WHERE profile = '{profile}'"
    )


def drop_hnsw_index_sql(profile: str) -> str:
    return f"DROP INDEX IF EXISTS {hnsw_index_name(profile)}"

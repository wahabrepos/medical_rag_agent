# db

Alembic migrations for the PostgreSQL + pgvector schema defined in `packages/db`.

```bash
docker compose -f deploy/docker-compose.dev.yml up -d        # local Postgres 17 + pgvector
uv run --env-file .env alembic -c db/alembic.ini upgrade head
```

import uuid

import pytest

from medrag_api.repository import SqlRunRepository
from medrag_db import make_engine, make_session_factory

pytestmark = pytest.mark.integration


def test_runs_and_feedback_round_trip(database_url: str) -> None:
    repo = SqlRunRepository(make_session_factory(make_engine(database_url)))
    run_id = uuid.uuid4()
    payload = {
        "question": "Q?",
        "answer": "B",
        "stop_reason": "stalled",
        "iterations": 2,
        "evidence": {"supported_fraction": 0.5, "status": "partially_supported"},
        "model": "m",
    }

    repo.save(run_id, payload)

    assert repo.get(run_id) == payload
    assert repo.get(uuid.uuid4()) is None
    assert repo.add_feedback(run_id, 1, "useful")
    assert not repo.add_feedback(uuid.uuid4(), -1, None)

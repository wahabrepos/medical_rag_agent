"""Where answered questions and feedback are stored."""

import uuid
from typing import Any, Protocol

from sqlalchemy.orm import Session, sessionmaker

from medrag_db.models import Feedback, Run


class RunRepository(Protocol):
    def save(self, run_id: uuid.UUID, response: dict[str, Any]) -> None: ...

    def get(self, run_id: uuid.UUID) -> dict[str, Any] | None: ...

    def add_feedback(self, run_id: uuid.UUID, rating: int, comment: str | None) -> bool: ...


class SqlRunRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def save(self, run_id: uuid.UUID, response: dict[str, Any]) -> None:
        evidence = response["evidence"]
        with self._sessions() as session, session.begin():
            session.add(
                Run(
                    id=run_id,
                    question=response["question"],
                    answer=response["answer"],
                    status="completed",
                    stop_reason=response["stop_reason"],
                    iterations=response["iterations"],
                    support_score=evidence["supported_fraction"],
                    model=response["model"],
                    payload=response,
                )
            )

    def get(self, run_id: uuid.UUID) -> dict[str, Any] | None:
        with self._sessions() as session:
            run = session.get(Run, run_id)
            return dict(run.payload) if run else None

    def add_feedback(self, run_id: uuid.UUID, rating: int, comment: str | None) -> bool:
        with self._sessions() as session, session.begin():
            if session.get(Run, run_id) is None:
                return False
            session.add(Feedback(run_id=run_id, rating=rating, comment=comment))
            return True


class InMemoryRunRepository:
    def __init__(self) -> None:
        self.runs: dict[uuid.UUID, dict[str, Any]] = {}
        self.feedback: list[tuple[uuid.UUID, int, str | None]] = []

    def save(self, run_id: uuid.UUID, response: dict[str, Any]) -> None:
        self.runs[run_id] = response

    def get(self, run_id: uuid.UUID) -> dict[str, Any] | None:
        return self.runs.get(run_id)

    def add_feedback(self, run_id: uuid.UUID, rating: int, comment: str | None) -> bool:
        if run_id not in self.runs:
            return False
        self.feedback.append((run_id, rating, comment))
        return True

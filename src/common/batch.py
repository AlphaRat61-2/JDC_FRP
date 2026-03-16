from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class BatchContext:
    batch_id: str
    run_type: str
    start_ts: datetime
    end_ts: datetime | None = None
    status: str = "RUNNING"
    notes: list[str] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)

    def add_note(self, msg: str) -> None:
        self.notes.append(msg)

    def set_row_count(self, table_name: str, count: int) -> None:
        self.row_counts[table_name] = int(count)


def create_batch_context(settings: dict, run_type: str, logger) -> BatchContext:
    batch_id = datetime.now().strftime("B%Y%m%d_%H%M%S")
    batch = BatchContext(
        batch_id=batch_id,
        run_type=run_type,
        start_ts=datetime.now(),
    )
    logger.info("Batch started | batch_id=%s | run_type=%s", batch.batch_id, batch.run_type)
    return batch


def close_batch_context(batch: BatchContext, logger, status: str) -> None:
    batch.end_ts = datetime.now()
    batch.status = status
    elapsed_sec = (batch.end_ts - batch.start_ts).total_seconds()
    logger.info(
        "Batch finished | batch_id=%s | status=%s | elapsed_sec=%.2f",
        batch.batch_id,
        batch.status,
        elapsed_sec,
    )
    if batch.row_counts:
        logger.info("Row counts | %s", batch.row_counts)
    if batch.notes:
        logger.info("Batch notes | %s", batch.notes)

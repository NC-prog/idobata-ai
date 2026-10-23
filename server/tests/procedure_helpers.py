"""テスト用の手続きナレッジ（WP09）。procedure_seeds.yaml の 3 件（承認済み）を一時ディレクトリに書く。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.config import SERVER_ROOT
from src.procedure_kb import ProcedureBook, load_seeds
from src.procedures import Procedures, load_procedures

SEEDS = SERVER_ROOT / "context" / "procedure_seeds.yaml"


def seeded_procedures(directory: Path | None = None, site: str = "etajima") -> Procedures:
    d = directory or Path(tempfile.mkdtemp(prefix="idobata-procs-"))
    book = ProcedureBook(d)
    for r in load_seeds(SEEDS, site):
        book.save(r)
    return load_procedures(SERVER_ROOT / "context", d)

"""手続きナレッジの出典の再確認（WP09 B-3）。知識パックの更新（kb/build.py）の後に動かす。

    cd server && uv run python ../scripts/check_procedures.py --site etajima

各手続きの ``sources[].url`` のページの本文ハッシュ（知識パックの page_hash）を、手続きの ``content_hash`` と比べ、
変わっていたら（またはパックから消えていたら）``status: recheck`` にする。職員パネルで「要再確認」として目立つ。
承認済みのまま使い続けるかは IDOBATA_PROCEDURE_RECHECK_USE（既定 1 = 使い続けて印だけ）。
下書きは印を付けない（承認の前に職員が出典を見るため）。外への通信はしない（パックを読むだけ）。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "server"))

from src.config import config
from src.procedure_kb import ProcedureBook, check_sources, procedures_dir


def pack_pages(data_dir: Path, site: str) -> dict[str, list[str]]:
    """知識パックの source_url → [ページの本文ハッシュ]。"""
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    table = pq.read_table((data_dir / "kb" / site).resolve() / "chunks.parquet", columns=["source_url", "page_hash"])
    return {u: [str(h or "")] for u, h in zip(table.column("source_url").to_pylist(),
                                                table.column("page_hash").to_pylist(), strict=True)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=config.site)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    book = ProcedureBook(procedures_dir(config.data_dir, args.site, config.procedure.subdir),
                         config.procedure.recheck_keep_using)
    changed = check_sources(book, pack_pages(config.data_dir, args.site))
    print(f"要再確認にした手続き: {len(changed)} 件 {' '.join(changed)}")


if __name__ == "__main__":
    main()

"""評価でお知らせ（WP11）を候補に入れる。記事は一時ディレクトリへの写しで読む（紹介回数を本物に書かない）。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from src.config import config
from src.news import NewsBook, load_rules, news_dir


def eval_news_book(site: str) -> NewsBook | None:
    src = news_dir(config.data_dir, site, config.news)
    if not config.news.enabled or not src.is_dir():
        return None
    tmp = Path(tempfile.mkdtemp(prefix="idobata-eval-news-")) / "news"
    shutil.copytree(src, tmp)
    return NewsBook(tmp, config.news, load_rules(config.news.rules_path))

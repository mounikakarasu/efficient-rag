from __future__ import annotations

import argparse
import time

from app.config import config
from app.logging_conf import configure_logging, get_logger
from app.qdrant_store import get_client, upload

log = get_logger("app.build_qdrant")


def main(argv: list[str] | None = None) -> None:
    configure_logging(config.log_level)
    parser = argparse.ArgumentParser(description="Upload the vector store into Qdrant.")
    parser.add_argument("--collection", default=config.qdrant_collection)
    parser.add_argument("--url", default=config.qdrant_url)
    args = parser.parse_args(argv)

    client = get_client(args.url)
    t0 = time.perf_counter()
    stats = upload(client, args.collection)
    elapsed = time.perf_counter() - t0
    print(f"Uploaded {stats['points']} points to Qdrant '{stats['collection']}' in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

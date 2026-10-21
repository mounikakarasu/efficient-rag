from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.config import config
from app.logging_conf import configure_logging, get_logger
from app.vector_index import build_flat, build_hnsw, build_ivfpq, load_vectors, save_index

log = get_logger("app.build_index")

BUILDERS = {"flat": build_flat, "hnsw": build_hnsw, "ivfpq": build_ivfpq}


def build_serving_index(embeddings_dir: Path, index_type: str) -> dict:
    vectors = load_vectors(embeddings_dir)
    t0 = time.perf_counter()
    index = BUILDERS[index_type](vectors)
    build_seconds = time.perf_counter() - t0
    path = embeddings_dir / "ann.index"
    save_index(index, path)
    stats = {
        "index_type": index_type,
        "vectors": int(vectors.shape[0]),
        "build_seconds": round(build_seconds, 2),
        "path": str(path),
    }
    log.info("serving index built", extra=stats)
    return stats


def main(argv: list[str] | None = None) -> None:
    configure_logging(config.log_level)
    parser = argparse.ArgumentParser(description="Build the serving ANN index from vectors.f32.")
    parser.add_argument("--type", default=config.ann_index_type, choices=list(BUILDERS))
    parser.add_argument("--emb", type=Path, default=config.embeddings_dir)
    args = parser.parse_args(argv)

    stats = build_serving_index(args.emb, args.type)
    print(
        f"Built {stats['index_type']} index over {stats['vectors']} vectors "
        f"in {stats['build_seconds']}s -> {stats['path']}"
    )


if __name__ == "__main__":
    main()

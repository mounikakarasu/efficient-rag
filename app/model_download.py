from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from app.config import config
from app.logging_conf import configure_logging, get_logger

log = get_logger("app.model_download")


def ensure_model(
    repo_id: str | None = None,
    filename: str | None = None,
    models_dir: Path | None = None,
) -> Path:
    repo_id = repo_id or config.llm_repo_id
    filename = filename or config.llm_filename
    models_dir = models_dir or config.models_dir
    target = models_dir / filename

    if target.exists():
        log.info("model already present", extra={"path": str(target)})
        return target

    models_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading model", extra={"repo_id": repo_id, "model_filename": filename})
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(models_dir),
    )
    log.info("model downloaded", extra={"path": downloaded})
    return Path(downloaded)


def main() -> None:
    configure_logging(config.log_level)
    path = ensure_model()
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Model ready: {path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()

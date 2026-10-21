from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw not in (None, "") else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return float(raw) if raw not in (None, "") else default


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    return Path(raw) if raw not in (None, "") else default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    data_dir: Path
    index_dir: Path
    models_dir: Path
    results_dir: Path
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    llm_repo_id: str
    llm_filename: str
    llm_n_ctx: int
    llm_n_threads: int
    llm_n_gpu_layers: int
    llm_max_tokens: int
    llm_temperature: float
    router_backend: str
    log_level: str

    corpus_dir: Path
    corpus_dataset_id: str
    corpus_split: str
    corpus_site_filter: str
    corpus_target_size: int
    eval_dir: Path
    eval_size: int

    embeddings_dir: Path
    embed_batch_size: int
    embed_batch_size_cuda: int
    embed_flush_size: int
    embedding_backend: str
    embedding_onnx_file: str
    embedding_device: str
    ann_index_type: str

    hybrid_enabled: bool
    rerank_enabled: bool
    reranker_model: str
    hybrid_candidate_k: int
    rrf_k: int
    rerank_top_n: int

    cache_enabled: bool
    cache_threshold: float
    cache_max_size: int

    vector_backend: str
    qdrant_url: str
    qdrant_collection: str

    offline_mode: bool
    redact_logs: bool
    auth_required: bool
    jwt_secret: str
    jwt_algorithm: str

    @property
    def llm_model_path(self) -> Path:
        override = os.environ.get("LLM_MODEL_PATH")
        if override not in (None, ""):
            return Path(override)
        return self.models_dir / self.llm_filename

    @property
    def index_path(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / "metadata.jsonl"


def load_config() -> Config:
    return Config(
        data_dir=_env_path("DATA_DIR", ROOT / "data"),
        index_dir=_env_path("INDEX_DIR", ROOT / "index"),
        models_dir=_env_path("MODELS_DIR", ROOT / "models"),
        results_dir=_env_path("RESULTS_DIR", ROOT / "results"),
        embedding_model=_env_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        chunk_size=_env_int("CHUNK_SIZE", 800),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 120),
        top_k=_env_int("TOP_K", 4),
        llm_repo_id=_env_str("LLM_REPO_ID", "Qwen/Qwen2.5-1.5B-Instruct-GGUF"),
        llm_filename=_env_str("LLM_FILENAME", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        llm_n_ctx=_env_int("LLM_N_CTX", 4096),
        llm_n_threads=_env_int("LLM_N_THREADS", 0),
        llm_n_gpu_layers=_env_int("LLM_N_GPU_LAYERS", 0),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 512),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.2),
        router_backend=_env_str("ROUTER_BACKEND", "plain"),
        log_level=_env_str("LOG_LEVEL", "INFO"),
        corpus_dir=_env_path("CORPUS_DIR", ROOT / "data" / "corpus"),
        corpus_dataset_id=_env_str("CORPUS_DATASET_ID", "HuggingFaceH4/stack-exchange-preferences"),
        corpus_split=_env_str("CORPUS_SPLIT", "train"),
        corpus_site_filter=_env_str("CORPUS_SITE_FILTER", "//stackoverflow.com"),
        corpus_target_size=_env_int("CORPUS_TARGET_SIZE", 2000),
        eval_dir=_env_path("EVAL_DIR", ROOT / "eval"),
        eval_size=_env_int("EVAL_SIZE", 200),
        embeddings_dir=_env_path("EMBEDDINGS_DIR", ROOT / "data" / "embeddings"),
        embed_batch_size=_env_int("EMBED_BATCH_SIZE", 64),
        embed_batch_size_cuda=_env_int("EMBED_BATCH_SIZE_CUDA", 128),
        embed_flush_size=_env_int("EMBED_FLUSH_SIZE", 2000),
        embedding_backend=_env_str("EMBEDDING_BACKEND", "onnx"),
        embedding_onnx_file=_env_str("EMBEDDING_ONNX_FILE", "onnx/model.onnx"),
        embedding_device=_env_str("EMBEDDING_DEVICE", "auto"),
        ann_index_type=_env_str("ANN_INDEX_TYPE", "hnsw"),
        hybrid_enabled=_env_bool("HYBRID_ENABLED", True),
        rerank_enabled=_env_bool("RERANK_ENABLED", True),
        reranker_model=_env_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        hybrid_candidate_k=_env_int("HYBRID_CANDIDATE_K", 50),
        rrf_k=_env_int("RRF_K", 60),
        rerank_top_n=_env_int("RERANK_TOP_N", 50),
        cache_enabled=_env_bool("CACHE_ENABLED", True),
        cache_threshold=_env_float("CACHE_THRESHOLD", 0.97),
        cache_max_size=_env_int("CACHE_MAX_SIZE", 256),
        vector_backend=_env_str("VECTOR_BACKEND", "faiss"),
        qdrant_url=_env_str("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=_env_str("QDRANT_COLLECTION", "efficient_rag"),
        offline_mode=_env_bool("OFFLINE_MODE", False),
        redact_logs=_env_bool("REDACT_LOGS", False),
        auth_required=_env_bool("AUTH_REQUIRED", False),
        jwt_secret=_env_str("JWT_SECRET", ""),
        jwt_algorithm=_env_str("JWT_ALGORITHM", "HS256"),
    )


config = load_config()

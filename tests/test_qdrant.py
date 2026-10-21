import numpy as np
import pytest


def _qdrant_available() -> bool:
    try:
        import httpx

        return httpx.get("http://localhost:6333/healthz", timeout=2).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _qdrant_available(), reason="Qdrant service not running")


def test_qdrant_roundtrip_and_filter():
    from qdrant_client.models import PointStruct

    from app.qdrant_store import create_collection, get_client
    from app.qdrant_store import search as qsearch

    client = get_client()
    name = "pytest_efficient_rag"
    create_collection(client, name, 8)
    try:
        vecs = np.eye(8, dtype="float32")
        points = [
            PointStruct(
                id=i,
                vector=vecs[i].tolist(),
                payload={"doc_id": str(i % 3), "text": f"chunk {i}", "chunk_index": 0, "url": f"u{i}"},
            )
            for i in range(8)
        ]
        client.upsert(collection_name=name, points=points, wait=True)

        top = qsearch(client, name, vecs[3], 1)
        assert int(top[0].id) == 3

        filtered = qsearch(client, name, vecs[0], 8, doc_id="1")
        assert filtered
        assert all((p.payload or {}).get("doc_id") == "1" for p in filtered)
    finally:
        client.delete_collection(name)

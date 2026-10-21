import app.main as main_module
from app.router import RouterResult
from fastapi.testclient import TestClient

client = TestClient(main_module.app)


def test_health_ok():
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_ask_calculate_end_to_end():
    resp = client.post("/ask", json={"question": "What is 6 * 7?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "calculate"
    assert "42" in body["answer"]
    assert body["latency_ms"] >= 0
    assert body["cached"] is False


def test_ask_semantic_cache_hit():
    q = {"question": "What is 8 * 9?"}
    first = client.post("/ask", json=q).json()
    second = client.post("/ask", json=q).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["answer"] == first["answer"]


def test_principal_from_headers_parsing():
    from app.access import Principal

    p = main_module._principal_from_headers("u1", "team-alpha, team-beta")
    assert isinstance(p, Principal)
    assert p.user_id == "u1"
    assert p.groups == frozenset({"team-alpha", "team-beta"})
    assert main_module._principal_from_headers(None, None) is None
    assert main_module._principal_from_headers("u", "") is None


def test_ask_with_principal_header(monkeypatch):
    monkeypatch.setattr(main_module, "get_retriever", lambda: object())
    resp = client.post(
        "/ask",
        json={"question": "What is 2 + 2?"},
        headers={"X-User-Groups": "team-alpha", "X-User-Id": "u1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "calculate"
    assert "4" in body["answer"]
    assert body["cached"] is False


def test_ask_bearer_token_authorizes(monkeypatch):
    from app.access import Principal

    monkeypatch.setattr(main_module, "get_retriever", lambda: object())
    monkeypatch.setattr(
        main_module, "principal_from_token", lambda tok: Principal("alice", frozenset({"team-alpha"}))
    )
    resp = client.post(
        "/ask",
        json={"question": "What is 2 + 2?"},
        headers={"Authorization": "Bearer abc.def.ghi"},
    )
    assert resp.status_code == 200
    assert resp.json()["cached"] is False


def test_ask_invalid_token_rejected(monkeypatch):
    from app.auth import AuthError

    def _raise(_tok):
        raise AuthError("invalid token")

    monkeypatch.setattr(main_module, "principal_from_token", _raise)
    resp = client.post("/ask", json={"question": "anything"}, headers={"Authorization": "Bearer bad"})
    assert resp.status_code == 401


def test_ask_stream_sse(monkeypatch):
    def fake_stream(_q, **_k):
        sources = [{"source": "http://x/1", "chunk_index": 0, "score": 0.9, "text": "ctx"}]
        return sources, iter(["Hello", " world"])

    monkeypatch.setattr(main_module, "answer_question_stream", fake_stream)
    with client.stream("POST", "/ask/stream", json={"question": "hi"}) as resp:
        body = "".join(resp.iter_text())
    assert "event: sources" in body
    assert "event: token" in body
    assert "Hello" in body and "world" in body
    assert "event: done" in body


def test_ask_validates_empty_question():
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_ask_returns_503_without_index(monkeypatch):
    def _boom(_question, **_kwargs):
        raise FileNotFoundError("No index found. Build one with app.ingest.")

    monkeypatch.setattr(main_module, "route_query", _boom)
    resp = client.post("/ask", json={"question": "anything"})
    assert resp.status_code == 503
    assert "index" in resp.json()["detail"].lower()


def test_ask_shape_with_sources(monkeypatch):
    canned = RouterResult(
        query="q", intent="retrieve_and_answer", answer="Venus.", tool="retriever+llm",
        sources=[{"source": "solar_system.txt", "chunk_index": 0, "score": 0.9, "text": "Venus is hot."}],
        latency_ms=123.4,
    )
    monkeypatch.setattr(main_module, "route_query", lambda _q, **_k: canned)
    body = client.post("/ask", json={"question": "hottest planet?"}).json()
    assert body["answer"] == "Venus."
    assert body["sources"][0]["source"] == "solar_system.txt"
    assert body["latency_ms"] == 123.4

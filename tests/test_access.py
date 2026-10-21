from app.access import TENANTS, Principal, SecureRetriever, doc_tenant
from app.retriever import RetrievedChunk


def test_doc_tenant_is_deterministic():
    assert doc_tenant("42") == doc_tenant("42")
    assert doc_tenant("42") in TENANTS


def test_principal_access():
    doc = "12345"
    owner = doc_tenant(doc)
    other = next(t for t in TENANTS if t != owner)
    assert Principal("u", frozenset({owner})).can_access(doc)
    assert not Principal("u", frozenset({other})).can_access(doc)


class _FakeBase:
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, query, top_k):
        return self._chunks[:top_k]


def test_secure_retriever_only_returns_permitted_docs():
    chunks = [
        RetrievedChunk(id=i, text=f"chunk {i}", source="", chunk_index=0, score=1.0 / (i + 1), doc_id=str(i))
        for i in range(60)
    ]
    target = TENANTS[0]
    retriever = SecureRetriever(_FakeBase(chunks), Principal("u", frozenset({target})), overfetch=10)
    results = retriever.search("q", top_k=5)
    assert results
    assert len(results) <= 5
    assert all(doc_tenant(c.doc_id) == target for c in results)


def test_secure_retriever_denies_when_no_group_match():
    chunks = [RetrievedChunk(id=1, text="x", source="", chunk_index=0, score=0.9, doc_id="1")]
    retriever = SecureRetriever(_FakeBase(chunks), Principal("u", frozenset({"team-none"})))
    assert retriever.search("q", top_k=5) == []

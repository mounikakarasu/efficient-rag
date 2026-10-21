from __future__ import annotations

from app.access import TENANTS, Principal, SecureRetriever, doc_tenant
from app.logging_conf import configure_logging
from app.offline import enforce_offline, missing_models, no_external_egress
from app.redaction import redact
from app.retriever import get_retriever

QUERY = "how do I read a file line by line in python"


def main() -> None:
    configure_logging("WARNING")

    print("== 1. Offline / no external egress ==")
    enforce_offline()
    missing = missing_models()
    print("missing local models:", missing or "none")
    retriever = get_retriever()
    retriever.search(QUERY, top_k=3)  # warm models at startup, before the guarded path
    with no_external_egress():
        hits = retriever.search(QUERY, top_k=3)
    print(f"retrieval under egress guard: {len(hits)} results, zero external connections")

    print("\n== 2. PII / secret redaction ==")
    sample = "Reach jane.doe@corp.com, SSN 123-45-6789, key AKIAIOSFODNN7EXAMPLE"
    print("original:", sample)
    print("redacted:", redact(sample))

    print("\n== 3. Access-controlled retrieval ==")
    for group in TENANTS:
        principal = Principal(user_id=f"user-{group}", groups=frozenset({group}))
        secure = SecureRetriever(get_retriever(), principal)
        results = secure.search(QUERY, top_k=5)
        tenants = sorted({doc_tenant(c.doc_id) for c in results})
        print(f"{group}: {len(results)} results, source tenants = {tenants}")


if __name__ == "__main__":
    main()

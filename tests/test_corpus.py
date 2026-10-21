import json

from app.corpus import (
    build_corpus,
    clean_html,
    normalize_record,
    normalized_stream,
    select_accepted_answer,
    site_matches,
)


def _rec(qid, question="<p>how to sort a list in python</p>", answers=None, site="https://stackoverflow.com/q"):
    return {
        "qid": str(qid),
        "question": question,
        "answers": answers if answers is not None else [],
        "metadata": [site],
    }


def test_clean_html():
    assert clean_html("<p>Hello &amp; <b>world</b></p>") == "Hello & world"


def test_select_accepted_answer_prefers_selected():
    answers = [
        {"selected": "False", "pm_score": "5", "text": "a"},
        {"selected": "True", "pm_score": "2", "text": "b"},
    ]
    assert select_accepted_answer(answers)["text"] == "b"


def test_select_accepted_answer_falls_back_to_score():
    answers = [
        {"selected": "False", "pm_score": "5", "text": "a"},
        {"selected": "False", "pm_score": "2", "text": "b"},
    ]
    assert select_accepted_answer(answers)["text"] == "a"
    assert select_accepted_answer([]) is None


def test_site_matches_excludes_meta():
    assert site_matches(_rec(1), "//stackoverflow.com")
    assert not site_matches(_rec(1, site="https://meta.stackoverflow.com/q"), "//stackoverflow.com")


def test_site_matches_is_case_insensitive():
    assert site_matches(_rec(1, site="https://Stackoverflow.com"), "//stackoverflow.com")


def test_normalize_record():
    rec = _rec(42, answers=[{"selected": "True", "pm_score": "9", "text": "<p>Use sorted()</p>"}])
    doc, ground_truth = normalize_record(rec)
    assert doc.id == "42"
    assert doc.source == "stackoverflow"
    assert "sort a list" in doc.text
    assert "Use sorted()" in doc.text
    assert ground_truth == "Use sorted()"


def test_normalized_stream_skip_and_limit():
    recs = [_rec(i) for i in range(10)]
    out = list(normalized_stream(recs, "//stackoverflow.com", limit=3, skip=2))
    assert [d.id for d, _ in out] == ["2", "3", "4"]


def test_normalized_stream_filters_site():
    recs = [_rec(1, site="https://meta.stackoverflow.com/q"), _rec(2)]
    out = list(normalized_stream(recs, "//stackoverflow.com", limit=10))
    assert [d.id for d, _ in out] == ["2"]


def test_build_corpus_writes_files(tmp_path, monkeypatch):
    recs = [_rec(i, answers=[{"selected": "True", "pm_score": "1", "text": f"<p>answer {i}</p>"}]) for i in range(5)]
    monkeypatch.setattr("app.corpus._load_stream", lambda *a, **k: recs)
    stats = build_corpus(tmp_path, "fake", "train", "//stackoverflow.com", target_size=5)
    assert stats["documents"] == 5
    assert len((tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()) == 5
    assert json.loads((tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()[0])["source"] == "stackoverflow"


def test_build_corpus_resumes(tmp_path, monkeypatch):
    recs = [_rec(i) for i in range(10)]
    monkeypatch.setattr("app.corpus._load_stream", lambda *a, **k: list(recs))
    build_corpus(tmp_path, "fake", "train", "//stackoverflow.com", target_size=3)
    assert len((tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    build_corpus(tmp_path, "fake", "train", "//stackoverflow.com", target_size=6)
    ids = [json.loads(line)["id"] for line in (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ids == ["0", "1", "2", "3", "4", "5"]

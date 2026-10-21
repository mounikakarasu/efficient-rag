from app import embeddings


def test_resolve_device_explicit():
    assert embeddings.resolve_device("cpu") == "cpu"
    assert embeddings.resolve_device("cuda") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert embeddings.resolve_device("auto") == "cpu"


def test_resolve_device_auto_uses_cuda_when_available(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert embeddings.resolve_device("auto") == "cuda"

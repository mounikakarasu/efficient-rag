import os
import socket

import pytest

from app.offline import EgressBlocked, enforce_offline, no_external_egress


def test_egress_guard_blocks_external():
    with no_external_egress():
        with pytest.raises(EgressBlocked):
            socket.create_connection(("8.8.8.8", 80), timeout=2)


def test_egress_guard_allows_local():
    with no_external_egress():
        with pytest.raises(OSError) as exc:
            socket.create_connection(("127.0.0.1", 65123), timeout=1)
        assert not isinstance(exc.value, EgressBlocked)


def test_enforce_offline_sets_env():
    prev_hf = os.environ.get("HF_HUB_OFFLINE")
    prev_tf = os.environ.get("TRANSFORMERS_OFFLINE")
    try:
        enforce_offline()
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    finally:
        for key, prev in (("HF_HUB_OFFLINE", prev_hf), ("TRANSFORMERS_OFFLINE", prev_tf)):
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev

from app.redaction import redact


def test_redacts_email():
    out = redact("contact jane.doe@example.com for access")
    assert "[EMAIL]" in out
    assert "jane.doe@example.com" not in out


def test_redacts_ssn_and_card():
    assert "[SSN]" in redact("ssn is 123-45-6789")
    assert "[CARD]" in redact("card 4111 1111 1111 1111 on file")


def test_redacts_keys_and_tokens():
    assert "[AWS_KEY]" in redact("AKIAIOSFODNN7EXAMPLE")
    assert "[TOKEN]" in redact("use sk-abcdefghijklmnopqrstuvwx now")
    assert "[JWT]" in redact("eyJhbGciOi.eyJzdWIiOi.SflKxwRJSM")


def test_leaves_ordinary_text_untouched():
    text = "how do I read a file line by line in python"
    assert redact(text) == text


def test_empty():
    assert redact("") == ""

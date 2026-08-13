"""Smoke test: package imports and exposes a version."""

import stonks


def test_version_is_string() -> None:
    assert isinstance(stonks.__version__, str)
    assert stonks.__version__


def test_version_format() -> None:
    # Expect a dotted version like "0.1.0".
    parts = stonks.__version__.split(".")
    assert len(parts) >= 2, f"version {stonks.__version__!r} should be dotted"

import pytest

from yieldcurve import config


def test_get_fred_api_key_reads_env(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "abc123")
    assert config.get_fred_api_key() == "abc123"


@pytest.mark.parametrize("value", ["", "your_key_here"])
def test_get_fred_api_key_missing_raises(monkeypatch, value):
    monkeypatch.setenv("FRED_API_KEY", value)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        config.get_fred_api_key()

import pyursa as pu
from pyursa import semantic as S


def test_set_llm_accepts_provider_string(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    before = S.CONFIG.client
    try:
        pu.set_llm("ollama:llama3.2")
        assert not isinstance(S.CONFIG.client, str)
        assert S.CONFIG.client.model == "llama3.2"
    finally:
        S.CONFIG.client = before

"""429 handling seen on a real Gemini free-tier key: wait as told, don't retry daily quotas, explain clearly."""

import io
import json
import urllib.error

import pytest

import pyursa as pu
from pyursa import providers as P


def _err(code, body, headers=None):
    return urllib.error.HTTPError("https://x", code, "Too Many Requests", headers or {}, io.BytesIO(json.dumps(body).encode()))


GEMINI_MINUTE = [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                            "message": "You exceeded your current quota, please check your plan and billing details. Please retry in 7.5s.",
                            "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                         "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
                                        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}]}}]
GEMINI_DAY = [{"error": {"code": 429, "message": "You exceeded your current quota, please check your plan and billing details.",
                         "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}]


def test_retry_delay_from_google_body():
    e = _err(429, GEMINI_MINUTE)
    assert "exceeded your current quota" in P._error_text(e)
    assert P._retry_after(e) == 7.0 and not P._daily_quota_exhausted(e)


def test_retry_after_header_wins():
    e = _err(429, GEMINI_MINUTE, {"Retry-After": "3"})
    P._error_text(e)
    assert P._retry_after(e) == 3.0


def test_daily_quota_fails_fast_with_clear_hint(monkeypatch):
    calls, sleeps = [], []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise _err(429, GEMINI_DAY)

    monkeypatch.setattr(P.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(P.time, "sleep", sleeps.append)
    c = P.OpenAICompatibleClient("gemini-x", provider="gemini", api_key="k", max_attempts=4)
    with pytest.raises(pu.LLMProviderError) as ei:
        c.complete("s", "hi")
    assert len(calls) == 1 and sleeps == []
    assert ei.value.details["status"] == 429
    assert "quota" in ei.value.suggestions[0]["description"] and "not a pyursa problem" in ei.value.suggestions[0]["description"]


def test_minute_limit_waits_as_told_then_succeeds(monkeypatch):
    sleeps, n = [], [0]

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout):
        n[0] += 1
        if n[0] == 1:
            raise _err(429, GEMINI_MINUTE)
        return Resp(json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {}}).encode())

    monkeypatch.setattr(P.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(P.time, "sleep", sleeps.append)
    c = P.OpenAICompatibleClient("gemini-x", provider="gemini", api_key="k", max_attempts=4)
    assert c.complete("s", "hi") == "ok" and sleeps == [7.0]

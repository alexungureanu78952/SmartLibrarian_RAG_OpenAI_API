from ragbot.openai_retry import call_with_retry


class TransientError(RuntimeError):
    pass


def test_call_with_retry_succeeds_after_retries(monkeypatch) -> None:
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            from openai import APITimeoutError

            raise APITimeoutError(request=None)
        return "ok"

    monkeypatch.setattr("time.sleep", lambda *_: None)
    assert call_with_retry(flaky, max_attempts=3) == "ok"
    assert attempts["count"] == 3

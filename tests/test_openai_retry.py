import pytest

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


def test_call_with_retry_rejects_non_callable_operation() -> None:
    with pytest.raises(TypeError):
        call_with_retry("not-callable")  # type: ignore[arg-type]


def test_call_with_retry_rejects_invalid_attempt_count() -> None:
    with pytest.raises(ValueError):
        call_with_retry(lambda: "ok", max_attempts=0)


def test_call_with_retry_rejects_negative_base_delay() -> None:
    with pytest.raises(ValueError):
        call_with_retry(lambda: "ok", base_delay=-0.1)

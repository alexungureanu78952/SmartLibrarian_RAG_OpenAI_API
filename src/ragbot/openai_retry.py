"""Retry helpers for transient OpenAI API failures."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError


T = TypeVar("T")


TRANSIENT_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


def call_with_retry(operation: Callable[[], T], max_attempts: int = 3, base_delay: float = 0.6) -> T:
    """Run an OpenAI API operation with bounded exponential backoff.

    Retries only transient network/service errors. Any non-transient exception is
    re-raised immediately.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except TRANSIENT_EXCEPTIONS as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            time.sleep(base_delay * (2 ** (attempt - 1)))
    assert last_error is not None
    raise last_error

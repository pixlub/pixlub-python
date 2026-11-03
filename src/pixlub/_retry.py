"""Retry policy: only requests that cannot cause a second charge are retried."""

from __future__ import annotations

import email.utils
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

INITIAL_BACKOFF = 0.5
MAX_BACKOFF = 8.0
MAX_RETRY_AFTER = 60.0
_RETRYABLE_GET_STATUSES = frozenset({500, 502, 503, 504})
_RETRYABLE_PUT_STATUSES = frozenset({500, 502, 503, 504, 507})
_IDEMPOTENT_METHODS = frozenset({"GET", "PUT"})

UPLOAD_INCOMPLETE_ATTEMPTS = 5
UPLOAD_INCOMPLETE_FIRST_PAUSE = 1.0


@dataclass(frozen=True)
class Failure:
    retryable: bool
    request_sent: bool
    timed_out: bool
    kind: str


def failure_for(exc: httpx.TransportError, method: str) -> Failure:
    timed_out = isinstance(exc, httpx.TimeoutException)
    never_sent = isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))
    return Failure(
        retryable=never_sent or method in _IDEMPOTENT_METHODS,
        request_sent=not never_sent,
        timed_out=timed_out,
        kind=type(exc).__name__,
    )


def status_is_retryable(method: str, status: int) -> bool:
    if status == 429:
        return True
    if method == "PUT":
        return status in _RETRYABLE_PUT_STATUSES
    return method == "GET" and status in _RETRYABLE_GET_STATUSES


def upload_incomplete_pause(attempt: int, retry_after: Optional[float] = None) -> float:
    if retry_after is not None:
        return min(max(retry_after, 0.0), MAX_RETRY_AFTER)
    pause: float = min(MAX_BACKOFF, UPLOAD_INCOMPLETE_FIRST_PAUSE * (2**attempt))
    return pause


def backoff_seconds(attempt: int, retry_after: Optional[float] = None) -> float:
    if retry_after is not None:
        return min(max(retry_after, 0.0), MAX_RETRY_AFTER)
    ceiling: float = min(MAX_BACKOFF, INITIAL_BACKOFF * (2**attempt))
    return ceiling * random.uniform(0.5, 1.0)  # noqa: S311


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when: datetime = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    seconds: float = (when - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, seconds)

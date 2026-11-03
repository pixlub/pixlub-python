"""Errors raised by the SDK. All derive from :class:`PixlubError`."""

from __future__ import annotations

from typing import Any, Dict, Optional


class PixlubError(Exception):
    """Base class of every error raised by the SDK."""

    job_id: Optional[str] = None
    """Id of the accepted run the error belongs to. The run may have been charged."""


class ConfigurationError(PixlubError, ValueError):
    """The API key or base URL is missing or invalid."""


class InvalidInputError(PixlubError, ValueError):
    """An argument was refused before anything was sent."""


class UnknownToolError(InvalidInputError):
    """The tool id is unknown."""


class APIConnectionError(PixlubError):
    """The API could not be reached, or the connection broke.

    When ``request_sent`` is True, the run may have been charged: check your
    balance before running it again.
    """

    def __init__(self, message: str, *, request_sent: bool) -> None:
        super().__init__(message)
        self.request_sent = request_sent


class APITimeoutError(APIConnectionError):
    """The API did not answer within the configured timeout."""


class JobTimeoutError(APITimeoutError):
    """The run did not finish within ``job_timeout``. It may still finish and was charged."""

    def __init__(self, message: str, *, job_id: str) -> None:
        super().__init__(message, request_sent=True)
        self.job_id = job_id


class UnexpectedResponseError(PixlubError):
    """The API answered with a success status, but not with the content the SDK expects."""


class APIError(PixlubError):
    """The API answered with an error status.

    The details are None when the server did not send them:

    - ``max_bytes``: largest file accepted (HTTP 413).
    - ``max_pixels``: largest image accepted, in pixels (``image_too_large``).
    - ``limit``: the limit that was reached (``batch_limit``, ``media_too_long``...).
    - ``required_credits``: credits the run needs (HTTP 402).
    - ``reset_at``: when a quota resets (ISO 8601), as sent by the server.
    - ``retry_after``: seconds to wait before trying again (429, 503), from the
      ``Retry-After`` header or the body.
    """

    def __init__(
        self,
        message: str,
        *,
        http_status: int,
        code: Optional[str] = None,
        details: Any = None,
        body: Optional[Dict[str, Any]] = None,
        retry_after: Optional[float] = None,
        max_bytes: Optional[int] = None,
        max_pixels: Optional[int] = None,
        limit: Optional[int] = None,
        required_credits: Optional[int] = None,
        reset_at: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.http_status = http_status
        self.code = code
        self.details = details
        self.body = body
        self.retry_after = retry_after
        self.max_bytes = max_bytes
        self.max_pixels = max_pixels
        self.limit = limit
        self.required_credits = required_credits
        self.reset_at = reset_at

    def __str__(self) -> str:
        suffix = f", code={self.code}" if self.code else ""
        return f"{self.message} (HTTP {self.http_status}{suffix})"


class InvalidRequestError(APIError):
    """HTTP 4xx: the request was refused, e.g. invalid parameters (see ``details``)."""


class FileTooLargeError(InvalidRequestError):
    """HTTP 413: a file is too large for the tool or your plan. ``max_bytes`` gives the limit."""


class AuthenticationError(APIError):
    """HTTP 401: the API key is missing, unknown or revoked."""


class PermissionDeniedError(APIError):
    """HTTP 403: the key is valid but the account may not do this, for example on a free plan."""


class NotFoundError(APIError):
    """HTTP 404: the tool, the upload or the job does not exist (see ``code``)."""


class InsufficientCreditsError(APIError):
    """HTTP 402: the account does not have enough credits for this tool."""


class RateLimitError(APIError):
    """HTTP 429: too many requests. ``retry_after`` is the wait in seconds, when given."""


class ServerError(APIError):
    """HTTP 5xx: something failed on the server side."""


class ServiceUnavailableError(ServerError):
    """HTTP 503: the service is busy or down for a while. ``retry_after`` says for how long."""

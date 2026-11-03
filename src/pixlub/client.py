"""Synchronous client for the Pixlub API."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union
from urllib.parse import quote

import httpx

from . import _core
from ._config import key_mask, resolve_api_key, resolve_base_url
from ._files import DEFAULT_MAX_INPUT_BYTES, Body, FileBody, PreparedFile
from ._retry import Failure, backoff_seconds, failure_for, status_is_retryable
from .errors import APIConnectionError, UnexpectedResponseError
from .models import Tool
from .tools import Tools

logger = logging.getLogger("pixlub")


class Client:
    """Client for the Pixlub API. Create one and reuse it; it is thread-safe.

    Args:
        api_key: Your API key. Defaults to ``PIXLUB_API_KEY``.
        base_url: Defaults to ``PIXLUB_BASE_URL``, then ``https://api.pixlub.com``.
        timeout: Seconds, or an ``httpx.Timeout``.
        max_retries: From 0 to 5. Defaults to 2.
        max_input_bytes: Largest local file accepted. Defaults to 3 GiB.
        job_timeout: Seconds to wait for a long run. Defaults to 1800.
        http_client: An ``httpx.Client`` to use, for proxies or custom TLS. Not closed by the SDK.

    Example::

        client = pixlub.Client()
        result = client.tools.run("removeBgApi", image="photo.jpg")
        result.save(f"photo-nobg{result.extension}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        max_retries: int = 2,
        max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
        job_timeout: float = _core.DEFAULT_JOB_TIMEOUT,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._api_key = resolve_api_key(api_key)
        self._base_url = resolve_base_url(base_url)
        self._timeout = _core.as_timeout(timeout)
        self._max_retries = _core.check_retries(max_retries)
        self._max_input_bytes = _core.check_positive(max_input_bytes, "max_input_bytes")
        self._job_timeout = _core.check_job_timeout(job_timeout)
        self._owns_http = http_client is None
        self._http = http_client if http_client is not None else httpx.Client()
        self._catalog_cache: Optional[_core.CatalogCache] = None
        self.tools = Tools(self)

    def close(self) -> None:
        """Close the connections, unless ``http_client`` was given."""
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Client(base_url={self._base_url!r}, api_key={key_mask(self._api_key)!r})"

    def _get_json(self, path: str) -> Any:
        response = self._execute(
            "GET", path, accept="application/json", max_bytes=_core.MAX_JSON_BYTES
        )
        return _core.json_body(response, self._api_key)

    def _catalog(self, *, refresh: bool = False) -> Tuple[Tool, ...]:
        now = time.monotonic()
        cached = None if refresh else _core.fresh_catalog(self._catalog_cache, now)
        if cached is not None:
            return cached
        payload = self._get_json("/v1/modules")
        if not isinstance(payload, list):
            raise UnexpectedResponseError("The tool catalog is not a list.")
        tools = tuple(_core.tool_from_payload(item) for item in payload)
        self._catalog_cache = (now, tools)
        return tools

    def _find_tool(self, tool_id: str) -> Optional[Tool]:
        cached = _core.fresh_catalog(self._catalog_cache, time.monotonic())
        tool = _core.find_tool(cached, tool_id) if cached is not None else None
        if tool is None:
            tool = _core.find_tool(self._catalog(refresh=True), tool_id)
        return tool

    def _execute(
        self,
        method: str,
        path: str,
        *,
        accept: str,
        max_bytes: int,
        timeout: Union[float, httpx.Timeout, None] = None,
        body: Optional[Body] = None,
        read_floor: Optional[float] = None,
    ) -> _core.Response:
        return self._request(
            method,
            self._base_url + path,
            headers=_core.request_headers(self._api_key, accept),
            label=path,
            max_bytes=max_bytes,
            timeout=timeout,
            body=body,
            read_floor=read_floor,
        )

    def _run_request(
        self, tool_id: str, body: Body, *, timeout: Union[float, httpx.Timeout, None]
    ) -> _core.Response:
        return self._execute(
            "POST",
            f"/v1/modules/{tool_id}",
            accept="application/json",
            max_bytes=_core.MAX_JSON_BYTES,
            timeout=timeout,
            body=body,
            read_floor=_core.RUN_READ_TIMEOUT_FLOOR,
        )

    def _open_upload(
        self, tool_id: str, file: PreparedFile, *, timeout: Union[float, httpx.Timeout, None]
    ) -> _core.UploadTicket:
        response = self._execute(
            "POST",
            "/v1/uploads",
            accept="application/json",
            max_bytes=_core.MAX_JSON_BYTES,
            timeout=timeout,
            body=_core.upload_request_body(tool_id, file),
        )
        answer = _core.upload_answer(response, self._api_key, self._base_url)
        while isinstance(answer, _core.UploadQueued):
            started = _core.clock()
            response = self._request(
                "GET",
                f"{self._base_url}/v1/uploads/{quote(answer.upload_id, safe='')}",
                headers=_core.upload_queue_headers(self._api_key),
                label="upload status",
                max_bytes=_core.MAX_JSON_BYTES,
                timeout=timeout,
                read_floor=_core.JOB_READ_TIMEOUT_FLOOR,
            )
            queued_id = answer.upload_id
            answer = _core.upload_answer(response, self._api_key, self._base_url)
            if answer.upload_id != queued_id:
                raise UnexpectedResponseError("The API answered for another upload.")
            if (
                isinstance(answer, _core.UploadQueued)
                and _core.clock() - started < _core.JOB_POLL_PAUSE
            ):
                time.sleep(_core.JOB_POLL_PAUSE)
        return answer

    def _put_upload(
        self,
        ticket: _core.UploadTicket,
        file: PreparedFile,
        *,
        timeout: Union[float, httpx.Timeout, None],
    ) -> bool:
        try:
            response = self._request(
                "PUT",
                ticket.url,
                headers=_core.download_headers(),
                label="file upload",
                max_bytes=_core.MAX_JSON_BYTES,
                timeout=timeout,
                body=FileBody(file),
                stop=ticket.expired,
            )
        except APIConnectionError:
            if ticket.expired():
                return False
            raise
        if _core.invalid_link(response) or (
            status_is_retryable("PUT", response.status) and ticket.expired()
        ):
            return False
        _core.check_put(response, self._api_key)
        return True

    def _follow_job(
        self,
        job_id: str,
        *,
        timeout: Union[float, httpx.Timeout, None],
        job_timeout: float,
    ) -> _core.Response:
        deadline = _core.clock() + job_timeout
        while True:
            remaining = deadline - _core.clock()
            if remaining <= 0:
                raise _core.job_timeout_error(job_id, job_timeout)
            started = _core.clock()
            response = self._request(
                "GET",
                f"{self._base_url}/v1/jobs/{job_id}",
                headers=_core.job_headers(self._api_key, remaining),
                label="job follow-up",
                max_bytes=_core.MAX_JSON_BYTES,
                timeout=timeout,
                read_floor=_core.JOB_READ_TIMEOUT_FLOOR,
            )
            if _core.job_finished(response, self._api_key, job_id):
                return response
            if _core.clock() - started < _core.JOB_POLL_PAUSE:
                time.sleep(max(0.0, min(_core.JOB_POLL_PAUSE, deadline - _core.clock())))

    def _download(
        self, url: str, *, timeout: Union[float, httpx.Timeout, None] = None
    ) -> _core.Response:
        return self._request(
            "GET",
            url,
            headers=_core.download_headers(),
            label="result download",
            max_bytes=_core.MAX_OUTPUT_BYTES,
            timeout=timeout,
        )

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Dict[str, str],
        label: str,
        max_bytes: int,
        timeout: Union[float, httpx.Timeout, None] = None,
        body: Optional[Body] = None,
        read_floor: Optional[float] = None,
        stop: Optional[Callable[[], bool]] = None,
    ) -> _core.Response:
        effective_timeout = _core.with_read_floor(
            self._timeout if timeout is None else _core.as_timeout(timeout), read_floor
        )
        attempt = 0
        while True:
            outcome = self._send_once(
                method,
                url,
                headers=headers,
                body=body,
                timeout=effective_timeout,
                max_bytes=max_bytes,
            )
            can_retry = attempt < self._max_retries and not (stop is not None and stop())
            if isinstance(outcome, Failure):
                if outcome.retryable and can_retry:
                    delay = backoff_seconds(attempt)
                    logger.debug(
                        "%s %s failed (%s), retrying in %.1fs", method, label, outcome.kind, delay
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise _core.connection_error(outcome)

            if status_is_retryable(method, outcome.status) and can_retry:
                retry_after = _core.retry_hint(outcome)
                delay = backoff_seconds(attempt, retry_after)
                logger.debug(
                    "%s %s returned HTTP %d, retrying in %.1fs",
                    method,
                    label,
                    outcome.status,
                    delay,
                )
                time.sleep(delay)
                attempt += 1
                continue
            return outcome

    def _send_once(
        self,
        method: str,
        url: str,
        *,
        headers: Dict[str, str],
        body: Optional[Body],
        timeout: httpx.Timeout,
        max_bytes: int,
    ) -> Union[_core.Response, Failure]:
        chunks = body.chunks() if body is not None else None
        if body is not None:
            headers = {
                **headers,
                "Content-Type": body.content_type,
                "Content-Length": str(body.length),
            }
        try:
            request = self._http.build_request(
                method, url, headers=headers, content=chunks, timeout=timeout
            )
            try:
                response = self._http.send(request, stream=True, follow_redirects=False)
            except httpx.TransportError as exc:
                return failure_for(exc, method)
            try:
                content, too_large = _read_limited(response, max_bytes)
            except httpx.TransportError as exc:
                return failure_for(exc, method)
            finally:
                response.close()
        finally:
            if chunks is not None:
                chunks.close()
        return _core.Response(
            status=response.status_code,
            headers=response.headers,
            body=content,
            too_large=too_large,
        )


def _read_limited(response: httpx.Response, limit: int) -> Tuple[bytes, bool]:
    buffer = bytearray()
    for chunk in response.iter_bytes():
        buffer.extend(chunk)
        if len(buffer) > limit:
            return b"", True
    return bytes(buffer), False

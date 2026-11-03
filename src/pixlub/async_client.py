"""Asynchronous client for the Pixlub API."""

from __future__ import annotations

import asyncio
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
from .tools import AsyncTools

logger = logging.getLogger("pixlub")


class AsyncClient:
    """Asynchronous client, with the same arguments and methods as :class:`pixlub.Client`::

    async with pixlub.AsyncClient() as client:
        result = await client.tools.run("removeBgApi", image="photo.jpg")
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
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._api_key = resolve_api_key(api_key)
        self._base_url = resolve_base_url(base_url)
        self._timeout = _core.as_timeout(timeout)
        self._max_retries = _core.check_retries(max_retries)
        self._max_input_bytes = _core.check_positive(max_input_bytes, "max_input_bytes")
        self._job_timeout = _core.check_job_timeout(job_timeout)
        self._owns_http = http_client is None
        self._http = http_client if http_client is not None else httpx.AsyncClient()
        self._catalog_cache: Optional[_core.CatalogCache] = None
        self.tools = AsyncTools(self)

    async def close(self) -> None:
        """Close the connections, unless ``http_client`` was given."""
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"AsyncClient(base_url={self._base_url!r}, api_key={key_mask(self._api_key)!r})"

    async def _get_json(self, path: str) -> Any:
        response = await self._execute(
            "GET", path, accept="application/json", max_bytes=_core.MAX_JSON_BYTES
        )
        return _core.json_body(response, self._api_key)

    async def _catalog(self, *, refresh: bool = False) -> Tuple[Tool, ...]:
        now = time.monotonic()
        cached = None if refresh else _core.fresh_catalog(self._catalog_cache, now)
        if cached is not None:
            return cached
        payload = await self._get_json("/v1/modules")
        if not isinstance(payload, list):
            raise UnexpectedResponseError("The tool catalog is not a list.")
        tools = tuple(_core.tool_from_payload(item) for item in payload)
        self._catalog_cache = (now, tools)
        return tools

    async def _find_tool(self, tool_id: str) -> Optional[Tool]:
        cached = _core.fresh_catalog(self._catalog_cache, time.monotonic())
        tool = _core.find_tool(cached, tool_id) if cached is not None else None
        if tool is None:
            tool = _core.find_tool(await self._catalog(refresh=True), tool_id)
        return tool

    async def _execute(
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
        return await self._request(
            method,
            self._base_url + path,
            headers=_core.request_headers(self._api_key, accept),
            label=path,
            max_bytes=max_bytes,
            timeout=timeout,
            body=body,
            read_floor=read_floor,
        )

    async def _run_request(
        self, tool_id: str, body: Body, *, timeout: Union[float, httpx.Timeout, None]
    ) -> _core.Response:
        return await self._execute(
            "POST",
            f"/v1/modules/{tool_id}",
            accept="application/json",
            max_bytes=_core.MAX_JSON_BYTES,
            timeout=timeout,
            body=body,
            read_floor=_core.RUN_READ_TIMEOUT_FLOOR,
        )

    async def _open_upload(
        self, tool_id: str, file: PreparedFile, *, timeout: Union[float, httpx.Timeout, None]
    ) -> _core.UploadTicket:
        response = await self._execute(
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
            response = await self._request(
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
                await asyncio.sleep(_core.JOB_POLL_PAUSE)
        return answer

    async def _put_upload(
        self,
        ticket: _core.UploadTicket,
        file: PreparedFile,
        *,
        timeout: Union[float, httpx.Timeout, None],
    ) -> bool:
        try:
            response = await self._request(
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

    async def _follow_job(
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
            response = await self._request(
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
                await asyncio.sleep(max(0.0, min(_core.JOB_POLL_PAUSE, deadline - _core.clock())))

    async def _download(
        self, url: str, *, timeout: Union[float, httpx.Timeout, None] = None
    ) -> _core.Response:
        return await self._request(
            "GET",
            url,
            headers=_core.download_headers(),
            label="result download",
            max_bytes=_core.MAX_OUTPUT_BYTES,
            timeout=timeout,
        )

    async def _request(
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
            outcome = await self._send_once(
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
                    await asyncio.sleep(delay)
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
                await asyncio.sleep(delay)
                attempt += 1
                continue
            return outcome

    async def _send_once(
        self,
        method: str,
        url: str,
        *,
        headers: Dict[str, str],
        body: Optional[Body],
        timeout: httpx.Timeout,
        max_bytes: int,
    ) -> Union[_core.Response, Failure]:
        chunks = body.achunks() if body is not None else None
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
                response = await self._http.send(request, stream=True, follow_redirects=False)
            except httpx.TransportError as exc:
                return failure_for(exc, method)
            try:
                content, too_large = await _read_limited(response, max_bytes)
            except httpx.TransportError as exc:
                return failure_for(exc, method)
            finally:
                await response.aclose()
        finally:
            if chunks is not None:
                await chunks.aclose()
        return _core.Response(
            status=response.status_code,
            headers=response.headers,
            body=content,
            too_large=too_large,
        )


async def _read_limited(response: httpx.Response, limit: int) -> Tuple[bytes, bool]:
    buffer = bytearray()
    async for chunk in response.aiter_bytes():
        buffer.extend(chunk)
        if len(buffer) > limit:
            return b"", True
    return bytes(buffer), False

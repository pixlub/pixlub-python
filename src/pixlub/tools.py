"""``client.tools``: list and run tools."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union

import httpx

from . import _core
from ._files import InputValue, PreparedFile, prepare_file
from ._retry import UPLOAD_INCOMPLETE_ATTEMPTS, upload_incomplete_pause
from .errors import InvalidInputError, PixlubError, UnexpectedResponseError
from .models import Tool, ToolResult

if TYPE_CHECKING:
    from .async_client import AsyncClient
    from .client import Client

Timeout = Union[float, httpx.Timeout, None]


class Tools:
    """List and run tools. Available as ``client.tools``."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def list(self, *, executable_only: bool = False) -> List[Tool]:
        """List the tools. ``executable_only`` leaves out those the API cannot run."""
        tools = self._client._catalog(refresh=True)
        if executable_only:
            return [tool for tool in tools if tool.executable]
        return list(tools)

    def retrieve(self, tool_id: str) -> Tool:
        """One tool. Raises ``UnknownToolError`` for an unknown id."""
        tool_id = _core.check_tool_id(tool_id)
        return _core.known_tool(tool_id, self._client._find_tool(tool_id))

    def schema(self, tool_id: str) -> Dict[str, Any]:
        """JSON Schema of a tool's ``params``."""
        tool_id = _core.check_tool_id(tool_id)
        payload = self._client._get_json(f"/v1/modules/{tool_id}/schema")
        if not isinstance(payload, dict):
            raise UnexpectedResponseError("The parameter schema is not a JSON object.")
        return payload

    def run(
        self,
        tool_id: str,
        *,
        image: Optional[InputValue] = None,
        inputs: Optional[Mapping[str, InputValue]] = None,
        params: Optional[Mapping[str, Any]] = None,
        timeout: Timeout = None,
        job_timeout: Optional[float] = None,
    ) -> ToolResult:
        """Run a tool and download what it produced.

        Args:
            tool_id: Such as ``"removeBgApi"``. See ``client.tools.list()``.
            image: The file of a tool with one input: a path, bytes, a binary
                file object or a ``pixlub.SourceURL``. A list for several files.
            inputs: Files keyed by input name, for tools with several inputs.
            params: The tool's settings. See ``client.tools.schema(tool_id)``.
            timeout: Overrides the client's timeout for this call.
            job_timeout: Overrides the client's ``job_timeout`` for this call.

        Large files are uploaded automatically, and long runs are awaited until
        ``job_timeout``. Arguments are checked before anything is sent.
        """
        client = self._client
        tool_id = _core.check_tool_id(tool_id)
        tool = _core.known_tool(tool_id, client._find_tool(tool_id))
        plan = _core.plan_run(tool_id, tool, image=image, inputs=inputs, params=params)
        job_timeout = (
            client._job_timeout if job_timeout is None else _core.check_job_timeout(job_timeout)
        )
        files: List[PreparedFile] = []
        try:
            for name, source in plan.files:
                files.append(
                    prepare_file(source, form_name=name, max_bytes=client._max_input_bytes)
                )
            transport = _core.plan_transport(plan, files)
            response = self._send(tool_id, plan, transport, timeout)
            direct_limit = _core.wants_upload(response) if transport.upload is None else None
            if direct_limit is not None:
                transport = _forced_upload(plan, files, direct_limit, response, client._api_key)
                response = self._send(tool_id, plan, transport, timeout)
        finally:
            for file in files:
                file.close()
        return self._finish(response, timeout, job_timeout)

    def _send(
        self, tool_id: str, plan: _core.RunPlan, transport: _core.Transport, timeout: Timeout
    ) -> _core.Response:
        client = self._client
        if transport.upload is None:
            return client._run_request(
                tool_id, _core.run_body(plan, transport.direct), timeout=timeout
            )

        upload_id = self._upload(tool_id, transport.upload, timeout)
        body = _core.run_body(plan, transport.direct, upload_id=upload_id)
        attempt = 0
        while True:
            response = client._run_request(tool_id, body, timeout=timeout)
            attempt += 1
            if not _core.upload_incomplete(response) or attempt >= UPLOAD_INCOMPLETE_ATTEMPTS:
                return response
            retry_after = _core.retry_hint(response)
            time.sleep(upload_incomplete_pause(attempt - 1, retry_after))

    def _upload(self, tool_id: str, file: PreparedFile, timeout: Timeout) -> str:
        for _ in range(2):
            ticket = self._client._open_upload(tool_id, file, timeout=timeout)
            if self._client._put_upload(ticket, file, timeout=timeout):
                return ticket.upload_id
        raise _link_expired_error()

    def _finish(self, response: _core.Response, timeout: Timeout, job_timeout: float) -> ToolResult:
        client = self._client
        job_id = (
            _core.accepted_job_id(response, client._api_key) if response.status == 202 else None
        )
        try:
            if job_id is not None:
                response = client._follow_job(job_id, timeout=timeout, job_timeout=job_timeout)
            receipt = _core.receipt_from_response(response, client._api_key, client._base_url)
        except PixlubError as error:
            _receipt_failed(error, response, job_id)
            raise
        try:
            outputs = [
                _core.output_from_response(
                    link, client._download(link.url, timeout=timeout), client._api_key
                )
                for link in receipt.outputs
            ]
        except PixlubError as error:
            if error.job_id is None:
                _core.after_acceptance(error, job_id=job_id, run_id=receipt.id)
            raise
        return _core.tool_result(receipt, outputs)


class AsyncTools:
    """Asynchronous version of :class:`Tools`."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    async def list(self, *, executable_only: bool = False) -> List[Tool]:
        """List the tools. See :meth:`Tools.list`."""
        tools = await self._client._catalog(refresh=True)
        if executable_only:
            return [tool for tool in tools if tool.executable]
        return list(tools)

    async def retrieve(self, tool_id: str) -> Tool:
        """One tool. See :meth:`Tools.retrieve`."""
        tool_id = _core.check_tool_id(tool_id)
        return _core.known_tool(tool_id, await self._client._find_tool(tool_id))

    async def schema(self, tool_id: str) -> Dict[str, Any]:
        """JSON Schema of a tool's ``params``. See :meth:`Tools.schema`."""
        tool_id = _core.check_tool_id(tool_id)
        payload = await self._client._get_json(f"/v1/modules/{tool_id}/schema")
        if not isinstance(payload, dict):
            raise UnexpectedResponseError("The parameter schema is not a JSON object.")
        return payload

    async def run(
        self,
        tool_id: str,
        *,
        image: Optional[InputValue] = None,
        inputs: Optional[Mapping[str, InputValue]] = None,
        params: Optional[Mapping[str, Any]] = None,
        timeout: Timeout = None,
        job_timeout: Optional[float] = None,
    ) -> ToolResult:
        """Run a tool. See :meth:`Tools.run`."""
        client = self._client
        tool_id = _core.check_tool_id(tool_id)
        tool = _core.known_tool(tool_id, await client._find_tool(tool_id))
        plan = _core.plan_run(tool_id, tool, image=image, inputs=inputs, params=params)
        job_timeout = (
            client._job_timeout if job_timeout is None else _core.check_job_timeout(job_timeout)
        )
        files: List[PreparedFile] = []
        try:
            for name, source in plan.files:
                files.append(
                    await asyncio.to_thread(
                        prepare_file, source, form_name=name, max_bytes=client._max_input_bytes
                    )
                )
            transport = _core.plan_transport(plan, files)
            response = await self._send(tool_id, plan, transport, timeout)
            direct_limit = _core.wants_upload(response) if transport.upload is None else None
            if direct_limit is not None:
                transport = _forced_upload(plan, files, direct_limit, response, client._api_key)
                response = await self._send(tool_id, plan, transport, timeout)
        finally:
            for file in files:
                file.close()
        return await self._finish(response, timeout, job_timeout)

    async def _send(
        self, tool_id: str, plan: _core.RunPlan, transport: _core.Transport, timeout: Timeout
    ) -> _core.Response:
        client = self._client
        if transport.upload is None:
            return await client._run_request(
                tool_id, _core.run_body(plan, transport.direct), timeout=timeout
            )

        upload_id = await self._upload(tool_id, transport.upload, timeout)
        body = _core.run_body(plan, transport.direct, upload_id=upload_id)
        attempt = 0
        while True:
            response = await client._run_request(tool_id, body, timeout=timeout)
            attempt += 1
            if not _core.upload_incomplete(response) or attempt >= UPLOAD_INCOMPLETE_ATTEMPTS:
                return response
            retry_after = _core.retry_hint(response)
            await asyncio.sleep(upload_incomplete_pause(attempt - 1, retry_after))

    async def _upload(self, tool_id: str, file: PreparedFile, timeout: Timeout) -> str:
        for _ in range(2):
            ticket = await self._client._open_upload(tool_id, file, timeout=timeout)
            if await self._client._put_upload(ticket, file, timeout=timeout):
                return ticket.upload_id
        raise _link_expired_error()

    async def _finish(
        self, response: _core.Response, timeout: Timeout, job_timeout: float
    ) -> ToolResult:
        client = self._client
        job_id = (
            _core.accepted_job_id(response, client._api_key) if response.status == 202 else None
        )
        try:
            if job_id is not None:
                response = await client._follow_job(
                    job_id, timeout=timeout, job_timeout=job_timeout
                )
            receipt = _core.receipt_from_response(response, client._api_key, client._base_url)
        except PixlubError as error:
            _receipt_failed(error, response, job_id)
            raise

        gate = asyncio.Semaphore(_core.DOWNLOAD_CONCURRENCY)

        async def fetch(link: _core.OutputLink) -> _core.Response:
            async with gate:
                return await client._download(link.url, timeout=timeout)

        tasks = [asyncio.ensure_future(fetch(link)) for link in receipt.outputs]
        try:
            downloads = await asyncio.gather(*tasks)
            outputs = [
                _core.output_from_response(link, download, client._api_key)
                for link, download in zip(receipt.outputs, downloads)
            ]
        except PixlubError as error:
            if error.job_id is None:
                _core.after_acceptance(error, job_id=job_id, run_id=receipt.id)
            raise
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return _core.tool_result(receipt, outputs)


def _receipt_failed(error: PixlubError, response: _core.Response, job_id: Optional[str]) -> None:
    if error.job_id is not None:
        return
    if job_id is not None:
        _core.after_acceptance(error, job_id=job_id, run_id=None)
    elif 200 <= response.status < 300:
        _core.after_acceptance(error, job_id=None, run_id=_core.run_id_of(response))


def _forced_upload(
    plan: _core.RunPlan,
    files: List[PreparedFile],
    direct_limit: int,
    refused: _core.Response,
    api_key: str,
) -> _core.Transport:
    try:
        return _core.plan_transport(
            plan, files, force_upload=True, direct_limit=direct_limit or None
        )
    except InvalidInputError:
        pass
    raise _core.error_from_response(refused, api_key)


def _link_expired_error() -> UnexpectedResponseError:
    return UnexpectedResponseError(
        "The upload link expired twice in a row. Check your clock, "
        "and that the file can be sent within 15 minutes."
    )

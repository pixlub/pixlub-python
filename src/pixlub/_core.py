"""Logic shared by the sync and async clients, without network I/O."""

from __future__ import annotations

import json
import math
import platform
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import httpx

from ._config import redact, resolve_download_url
from ._files import (
    Body,
    BytesBody,
    FileSource,
    InputValue,
    MultipartBody,
    PreparedFile,
    SourceURL,
)
from ._retry import Failure, parse_retry_after
from ._version import __version__
from .errors import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    ConfigurationError,
    FileTooLargeError,
    InsufficientCreditsError,
    InvalidInputError,
    InvalidRequestError,
    JobTimeoutError,
    NotFoundError,
    PermissionDeniedError,
    PixlubError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    UnexpectedResponseError,
    UnknownToolError,
)
from .models import Tool, ToolInput, ToolOutput, ToolOutputSpec, ToolResult

USER_AGENT = f"pixlub-python/{__version__} (Python {platform.python_version()})"

DEFAULT_TIMEOUT = httpx.Timeout(10.0, read=180.0, write=60.0)
RUN_READ_TIMEOUT_FLOOR = 70.0
JOB_READ_TIMEOUT_FLOOR = 35.0
DEFAULT_JOB_TIMEOUT = 1800.0

DIRECT_UPLOAD_MAX_BYTES = 90 * 1024 * 1024
MAX_INPUT_FILES = 64
MAX_OUTPUTS = MAX_INPUT_FILES * 4
# Largest output read into memory: above any image the API produces (uploaded
# images go up to 1 GiB). MAX_RUN_OUTPUT_BYTES: all the outputs of one run together.
MAX_OUTPUT_BYTES = 1024 * 1024 * 1024
MAX_RUN_OUTPUT_BYTES = 4 * 1024 * 1024 * 1024
MAX_JSON_BYTES = 10 * 1024 * 1024
CATALOG_TTL_SECONDS = 600.0
MAX_RETRIES_LIMIT = 5
DOWNLOAD_CONCURRENCY = 4

JOB_WAIT_SECONDS = 25
JOB_POLL_PAUSE = 2.0

UPLOAD_QUEUE_WAIT_SECONDS = 25

_JOB_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_UPLOAD_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_TOOL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_FIELD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_FRACTION_RE = re.compile(r"(\.\d{1,6})\d*")


def clock() -> float:
    return time.monotonic()


def check_tool_id(tool_id: object) -> str:
    if not isinstance(tool_id, str) or not _TOOL_ID_RE.fullmatch(tool_id):
        raise InvalidInputError(
            f"Invalid tool id {str(tool_id)[:80]!r}. Ids use letters, digits, '-' and '_', "
            "such as 'removeBgApi'."
        )
    return tool_id


def request_headers(api_key: str, accept: str) -> Dict[str, str]:
    return {"X-API-Key": api_key, "User-Agent": USER_AGENT, "Accept": accept}


def download_headers() -> Dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "*/*"}


@dataclass(frozen=True)
class Response:
    status: int
    headers: httpx.Headers
    body: bytes
    too_large: bool = False


def tool_from_payload(item: Any) -> Tool:
    if not isinstance(item, dict) or not isinstance(item.get("id"), str):
        raise UnexpectedResponseError("The tool catalog contains an entry without an id.")
    raw_inputs = item.get("inputs")
    inputs = tuple(
        ToolInput(
            id=entry["id"],
            label=_text(entry.get("label")),
            required=entry.get("required") is True,
            type=_text(entry.get("type")) or "image",
            description=_text(entry.get("description")),
            multiple=entry.get("multiple") is True,
            per_file=entry.get("per_file") is True,
            max_files=_positive_int(entry.get("max_files")),
            accept=_strings(entry.get("accept")),
        )
        for entry in (raw_inputs if isinstance(raw_inputs, list) else [])
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    )
    raw_outputs = item.get("outputs")
    outputs = tuple(
        ToolOutputSpec(
            key=entry["key"],
            label=_text(entry.get("label")),
            description=_text(entry.get("description")),
            kind=_text(entry.get("kind")),
            multiple=entry.get("multiple") is True,
            content_types=_strings(entry.get("content_types")),
        )
        for entry in (raw_outputs if isinstance(raw_outputs, list) else [])
        if isinstance(entry, dict) and isinstance(entry.get("key"), str)
    )
    return Tool(
        id=item["id"],
        name=_text(item.get("name")) or item["id"],
        description=_text(item.get("description")),
        category=_text(item.get("category")),
        executable=item.get("executable") is True,
        handler=_text(item.get("handler")) or "none",
        inputs=inputs,
        outputs=outputs,
    )


CatalogCache = Tuple[float, Tuple[Tool, ...]]


def fresh_catalog(cache: Optional[CatalogCache], now: float) -> Optional[Tuple[Tool, ...]]:
    if cache is not None and now - cache[0] < CATALOG_TTL_SECONDS:
        return cache[1]
    return None


def find_tool(tools: Sequence[Tool], tool_id: str) -> Optional[Tool]:
    for tool in tools:
        if tool.id == tool_id:
            return tool
    return None


def known_tool(tool_id: str, tool: Optional[Tool]) -> Tool:
    if tool is None:
        raise UnknownToolError(
            f"'{tool_id}' is not in the tool catalog. Call client.tools.list() to see the ids."
        )
    return tool


@dataclass(frozen=True)
class RunPlan:
    files: Tuple[Tuple[str, FileSource], ...]
    params_json: str
    source_url: Optional[str] = None
    primary: Optional[str] = None


def plan_run(
    tool_id: str,
    tool: Tool,
    *,
    image: Optional[InputValue],
    inputs: Optional[Mapping[str, InputValue]],
    params: Optional[Mapping[str, Any]],
) -> RunPlan:
    if not tool.executable:
        raise InvalidInputError(f"'{tool_id}' cannot be run through the API.")
    if image is not None and inputs is not None:
        raise InvalidInputError("Pass either image= or inputs=, not both.")

    named: List[Tuple[str, List[Any]]]
    if image is not None:
        named = [(_single_image_input(tool), _as_list("image=", image))]
    elif inputs is not None:
        named = _named_inputs(tool_id, tool, inputs)
    else:
        named = []
        required = [i.id for i in tool.inputs if i.required]
        if required:
            raise InvalidInputError(
                f"'{tool_id}' needs a file for: {', '.join(required)}. "
                "Pass image= or inputs={...}."
            )

    main = tool.primary_input
    primary = main.id if main is not None else None

    files: List[Tuple[str, FileSource]] = []
    source_url: Optional[str] = None
    specs = {i.id: i for i in tool.inputs}
    for name, values in named:
        urls = [v for v in values if isinstance(v, SourceURL)]
        if urls:
            if len(values) != 1 or source_url is not None:
                raise InvalidInputError("A run takes one SourceURL, alone in its input.")
            if name != primary:
                raise InvalidInputError(
                    f"A SourceURL can only fill '{primary}', the main input of '{tool_id}'."
                )
            source_url, primary = urls[0].url, name
            continue
        spec = specs[name]
        if not spec.multiple and len(values) > 1:
            raise InvalidInputError(f"Input '{name}' of '{tool_id}' takes one file, not a list.")
        if spec.max_files is not None and len(values) > spec.max_files:
            raise InvalidInputError(
                f"Input '{name}' of '{tool_id}' takes at most {spec.max_files} files."
            )
        files.extend((name, value) for value in values)

    if len(files) > MAX_INPUT_FILES:
        raise InvalidInputError(f"A run sends at most {MAX_INPUT_FILES} files.")
    return RunPlan(
        files=tuple(files),
        params_json=encode_params(params),
        source_url=source_url,
        primary=primary,
    )


def _as_list(label: str, value: Any) -> List[Any]:
    # Only lists and tuples hold several files: bytes and str are sequences too.
    if isinstance(value, (list, tuple)):
        if not value:
            raise InvalidInputError(f"{label} got an empty list.")
        return list(value)
    return [value]


def _single_image_input(tool: Tool) -> str:
    if not tool.inputs:
        raise InvalidInputError(f"'{tool.id}' takes no image. Pass its settings with params= only.")
    required = [i for i in tool.inputs if i.required]
    candidates = required or list(tool.inputs)
    if len(candidates) == 1:
        return candidates[0].id
    names = ", ".join(i.id for i in tool.inputs)
    raise InvalidInputError(
        f"'{tool.id}' takes several images ({names}). Pass inputs={{'<name>': image, ...}}."
    )


def _named_inputs(
    tool_id: str, tool: Tool, inputs: Mapping[str, InputValue]
) -> List[Tuple[str, List[Any]]]:
    if not isinstance(inputs, Mapping) or not inputs:
        raise InvalidInputError("inputs= must be a non-empty dict such as {'image': 'photo.png'}.")
    known = {i.id for i in tool.inputs}
    named: List[Tuple[str, List[Any]]] = []
    for name, value in inputs.items():
        if not isinstance(name, str) or not _FIELD_RE.fullmatch(name):
            raise InvalidInputError(f"Invalid input name {str(name)[:80]!r}.")
        if name not in known:
            available = ", ".join(sorted(known)) or "none"
            raise InvalidInputError(
                f"'{tool_id}' has no input named '{name}'. Its inputs are: {available}."
            )
        named.append((name, _as_list(f"Input '{name}'", value)))
    missing = [i.id for i in tool.inputs if i.required and i.id not in inputs]
    if missing:
        raise InvalidInputError(f"'{tool_id}' is missing required input(s): {', '.join(missing)}.")
    return named


def encode_params(params: Optional[Mapping[str, Any]]) -> str:
    if params is None:
        return "{}"
    if not isinstance(params, Mapping):
        raise InvalidInputError("params= must be a dict such as {'width': 512}.")
    try:
        return json.dumps(dict(params), allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"params= must contain only JSON values: {exc}.") from None


@dataclass(frozen=True)
class Transport:
    upload: Optional[PreparedFile]
    direct: Tuple[PreparedFile, ...]


def plan_transport(
    plan: RunPlan,
    files: Sequence[PreparedFile],
    *,
    force_upload: bool = False,
    direct_limit: Optional[int] = None,
) -> Transport:
    limit = (
        DIRECT_UPLOAD_MAX_BYTES
        if direct_limit is None
        else min(direct_limit, DIRECT_UPLOAD_MAX_BYTES)
    )
    total = sum(f.size for f in files)
    if total <= limit and not force_upload:
        return Transport(upload=None, direct=tuple(files))

    too_much = (
        f"The files of this run total {total} bytes, more than the {limit} bytes "
        "one request can carry"
    )
    if plan.source_url is not None:
        raise InvalidInputError(
            f"{too_much}. The SourceURL fills the main input, so the other files must fit."
        )
    if plan.primary is None:
        raise InvalidInputError(
            f"{too_much}, and this tool has no main input, so none can be uploaded on its "
            "own. Send a smaller file."
        )
    mine = [f for f in files if f.form_name == plan.primary]
    others = [f for f in files if f.form_name != plan.primary]
    if not mine:
        raise InvalidInputError(
            f"{too_much}. Only the main input, '{plan.primary}', can be uploaded on its own."
        )
    if len(mine) > 1:
        raise InvalidInputError(
            f"{too_much}. An upload carries one file, and '{plan.primary}' got {len(mine)}: "
            "split them into several runs."
        )
    rest = sum(f.size for f in others)
    if rest > limit:
        raise InvalidInputError(
            f"{too_much}. Even with '{plan.primary}' uploaded on its own, the other inputs "
            f"total {rest} bytes."
        )
    return Transport(upload=mine[0], direct=tuple(others))


def run_body(
    plan: RunPlan, files: Sequence[PreparedFile], *, upload_id: Optional[str] = None
) -> MultipartBody:
    fields: List[Tuple[str, str]] = [("params", plan.params_json)]
    if upload_id is not None:
        fields.append(("upload_id", upload_id))
    elif plan.source_url is not None:
        fields.append(("source_url", plan.source_url))
    return MultipartBody(fields, files)


def upload_request_body(tool_id: str, file: PreparedFile) -> Body:
    payload = {"module": tool_id, "filename": file.filename, "bytes": file.size}
    return BytesBody(json.dumps(payload).encode(), "application/json")


def wants_upload(resp: Response) -> Optional[int]:
    payload = _json_object(resp.body) if resp.status == 413 else None
    if payload is None or payload.get("code") != "use_uploads":
        return None
    return _positive_int(payload.get("max_bytes")) or 0


def upload_incomplete(resp: Response) -> bool:
    if resp.status != 409:
        return False
    payload = _json_object(resp.body)
    return payload is not None and payload.get("code") == "upload_incomplete"


def invalid_link(resp: Response) -> bool:
    if resp.status != 403:
        return False
    payload = _json_object(resp.body)
    return payload is not None and payload.get("code") == "invalid_link"


@dataclass(frozen=True)
class UploadTicket:
    upload_id: str
    url: str
    expires_at: Optional[float]

    def expired(self) -> bool:
        return self.expires_at is not None and time.time() >= self.expires_at


@dataclass(frozen=True)
class UploadQueued:
    upload_id: str
    position: int


def upload_answer(resp: Response, api_key: str, base_url: str) -> Union[UploadTicket, UploadQueued]:
    payload, upload_id = _upload_payload(resp, api_key)
    if payload.get("status") == "queued":
        return UploadQueued(
            upload_id=upload_id, position=_positive_int(payload.get("position")) or 0
        )
    return _upload_ticket(payload, upload_id, base_url)


def upload_ticket(resp: Response, api_key: str, base_url: str) -> UploadTicket:
    payload, upload_id = _upload_payload(resp, api_key)
    return _upload_ticket(payload, upload_id, base_url)


def upload_queue_headers(api_key: str) -> Dict[str, str]:
    headers = request_headers(api_key, "application/json")
    headers["Prefer"] = f"wait={UPLOAD_QUEUE_WAIT_SECONDS}"
    return headers


def _upload_payload(resp: Response, api_key: str) -> Tuple[Dict[str, Any], str]:
    payload = json_body(resp, api_key)
    if not isinstance(payload, dict):
        raise UnexpectedResponseError("The upload answer is not a JSON object.")
    upload_id = payload.get("upload_id")
    if not isinstance(upload_id, str) or not _UPLOAD_ID_RE.fullmatch(upload_id):
        raise UnexpectedResponseError("The upload answer has no valid upload id.")
    return payload, upload_id


def _upload_ticket(payload: Dict[str, Any], upload_id: str, base_url: str) -> UploadTicket:
    method = payload.get("method", "PUT")
    if method != "PUT":
        raise UnexpectedResponseError("The upload answer asks for a method other than PUT.")
    link = payload.get("upload_url")
    try:
        url = resolve_download_url(link if isinstance(link, str) else "", base_url)
    except ValueError as exc:
        raise UnexpectedResponseError(f"Refused an upload link: {exc}.") from None
    return UploadTicket(
        upload_id=upload_id, url=url, expires_at=parse_timestamp(payload.get("expires_at"))
    )


def parse_timestamp(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not value:
        return None
    # Before Python 3.11, fromisoformat rejects "Z" and fractions other than 3 or 6 digits.
    text = _FRACTION_RE.sub(
        lambda match: match.group(1).ljust(7, "0"),
        value.strip().upper().replace("Z", "+00:00"),
        1,
    )
    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.timestamp()


def error_from_response(resp: Response, api_key: str) -> APIError:
    payload = _json_object(resp.body)
    message: Optional[str] = None
    code: Optional[str] = None
    details: Any = None
    if payload is not None:
        for key in ("message", "error"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                message = value.strip()
                break
        if isinstance(payload.get("code"), str):
            code = payload["code"]
        details = payload.get("details")
    body = payload or {}

    status = resp.status
    text = redact(message or f"The Pixlub API returned HTTP {status}.", api_key)[:500]
    cls: Type[APIError]
    if status == 401:
        cls = AuthenticationError
    elif status == 402:
        cls = InsufficientCreditsError
    elif status == 403:
        cls = PermissionDeniedError
    elif status == 404:
        cls = NotFoundError
    elif status == 413:
        cls = FileTooLargeError
    elif status == 429:
        cls = RateLimitError
    elif 400 <= status < 500:
        cls = InvalidRequestError
    elif status == 503:
        cls = ServiceUnavailableError
    elif 500 <= status < 600:
        cls = ServerError
    else:
        cls = APIError

    retry_after = parse_retry_after(resp.headers.get("retry-after"))
    if retry_after is None:
        retry_after = _seconds(body.get("retry_after"))
    reset_at = body.get("reset_at")
    return cls(
        text,
        http_status=status,
        code=code,
        details=details,
        body=payload,
        retry_after=retry_after,
        max_bytes=_positive_int(body.get("max_bytes")),
        max_pixels=_positive_int(body.get("max_pixels")),
        limit=_as_int(body.get("limit")),
        required_credits=_as_int(body.get("required")),
        reset_at=reset_at[:64] if isinstance(reset_at, str) else None,
    )


def retry_hint(resp: Response) -> Optional[float]:
    hint = parse_retry_after(resp.headers.get("retry-after"))
    if hint is None:
        payload = _json_object(resp.body) if len(resp.body) <= 64 * 1024 else None
        hint = _seconds(payload.get("retry_after")) if payload is not None else None
    return hint


def json_body(resp: Response, api_key: str) -> Any:
    if resp.too_large:
        raise UnexpectedResponseError("The API returned more data than the SDK accepts.")
    if not 200 <= resp.status < 300:
        raise error_from_response(resp, api_key)
    try:
        return json.loads(resp.body)
    except ValueError:
        raise UnexpectedResponseError("The API returned a body that is not valid JSON.") from None


def check_put(resp: Response, api_key: str) -> None:
    if not 200 <= resp.status < 300:
        raise error_from_response(resp, api_key)


def accepted_job_id(resp: Response, api_key: str) -> str:
    payload = json_body(resp, api_key)
    job_id = payload.get("job_id") if isinstance(payload, dict) else None
    if not isinstance(job_id, str) or not _JOB_ID_RE.fullmatch(job_id):
        raise UnexpectedResponseError("The API accepted the run without a job id to follow.")
    return job_id


def job_headers(api_key: str, remaining: float) -> Dict[str, str]:
    headers = request_headers(api_key, "application/json")
    wait = max(1, min(JOB_WAIT_SECONDS, math.ceil(remaining)))
    headers["Prefer"] = f"wait={wait}"
    return headers


def job_finished(resp: Response, api_key: str, job_id: str) -> bool:
    payload = json_body(resp, api_key)
    if not isinstance(payload, dict):
        raise UnexpectedResponseError("The job answer is not a JSON object.")
    status = payload.get("status")
    if status == "succeeded":
        return True
    if status in ("queued", "running"):
        return False
    if status == "failed":
        http_status = payload.get("http_status")
        if not isinstance(http_status, int) or isinstance(http_status, bool) or http_status < 400:
            http_status = 500
        error = error_from_response(
            Response(status=http_status, headers=resp.headers, body=resp.body), api_key
        )
        error.job_id = job_id
        error.message = f"{error.message} (job {job_id})"
        error.args = (error.message,)
        raise error
    raise UnexpectedResponseError("The job answer has an unknown status.")


def job_timeout_error(job_id: str, seconds: float) -> JobTimeoutError:
    return JobTimeoutError(
        f"Job {job_id} was still running after {seconds:g} s (job_timeout). It was accepted, "
        "so its credits may have been charged, and it may still finish: follow it with "
        f"GET /v1/jobs/{job_id} rather than running the tool again.",
        job_id=job_id,
    )


def after_acceptance(error: PixlubError, *, job_id: Optional[str], run_id: Optional[str]) -> None:
    if job_id is not None:
        error.job_id = job_id
        note = (
            f"Job {job_id} was accepted, so its credits may have been charged: follow it with "
            f"GET /v1/jobs/{job_id} before running the tool again."
        )
    else:
        which = f"Run {run_id}" if run_id else "The run"
        note = f"{which} succeeded and was charged: do not run it again to get its result."
    if isinstance(error, APIError):
        error.message = f"{error.message} {note}"
        error.args = (error.message,)
    else:
        error.args = (f"{error.args[0] if error.args else ''} {note}".strip(),)


@dataclass(frozen=True)
class OutputLink:
    name: str
    url: str
    content_type: str
    key: str = ""
    bytes: Optional[int] = None


@dataclass(frozen=True)
class RunReceipt:
    module: str
    outputs: Tuple[OutputLink, ...]
    id: Optional[str]
    expires_at: Optional[str]
    cost: Optional[int]
    credits_balance: Optional[int]
    message: Optional[str]


def receipt_from_response(resp: Response, api_key: str, base_url: str) -> RunReceipt:
    payload = json_body(resp, api_key)
    if not isinstance(payload, dict):
        raise UnexpectedResponseError("The run response is not a JSON object.")

    raw_outputs = payload.get("outputs")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise UnexpectedResponseError("The run response lists no output.")
    if len(raw_outputs) > MAX_OUTPUTS:
        raise UnexpectedResponseError(
            f"The run response lists {len(raw_outputs)} outputs; the SDK reads at most "
            f"{MAX_OUTPUTS}."
        )
    outputs: List[OutputLink] = []
    for entry in raw_outputs:
        if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
            raise UnexpectedResponseError("The run response contains an output without a link.")
        try:
            url = resolve_download_url(entry["url"], base_url)
        except ValueError as exc:
            raise UnexpectedResponseError(f"Refused a result link: {exc}.") from None
        outputs.append(
            OutputLink(
                name=_text(entry.get("name")) or "file",
                url=url,
                content_type=_text(entry.get("content_type")) or "application/octet-stream",
                key=_text(entry.get("key")),
                bytes=_as_int(entry.get("bytes")),
            )
        )

    # Announced sizes are checked before anything is downloaded.
    for link in outputs:
        if link.bytes is not None and link.bytes > MAX_OUTPUT_BYTES:
            raise UnexpectedResponseError(
                f"One of its files is {link.bytes} bytes, more than the {MAX_OUTPUT_BYTES} bytes "
                "the SDK downloads: get it from its link instead."
            )
    if sum(link.bytes or 0 for link in outputs) > MAX_RUN_OUTPUT_BYTES:
        raise UnexpectedResponseError("Its results are larger than the SDK accepts.")

    credits = payload.get("credits")
    credits = credits if isinstance(credits, dict) else {}
    message = payload.get("message")
    return RunReceipt(
        module=_text(payload.get("module")),
        outputs=tuple(outputs),
        id=payload["id"] if isinstance(payload.get("id"), str) else None,
        expires_at=payload["expires_at"] if isinstance(payload.get("expires_at"), str) else None,
        cost=_as_int(credits.get("cost")),
        credits_balance=_as_int(credits.get("balance")),
        message=message[:1000] if isinstance(message, str) else None,
    )


def run_id_of(resp: Response) -> Optional[str]:
    payload = _json_object(resp.body)
    run_id = payload.get("id") if payload is not None else None
    return run_id[:128] if isinstance(run_id, str) and run_id else None


def output_from_response(link: OutputLink, resp: Response, api_key: str) -> ToolOutput:
    if resp.too_large:
        raise UnexpectedResponseError(
            f"The tool output is larger than {MAX_OUTPUT_BYTES} bytes and was not read."
        )
    if not 200 <= resp.status < 300:
        raise error_from_response(resp, api_key)
    content_type = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    return ToolOutput(
        name=link.name,
        url=link.url,
        content_type=content_type or link.content_type,
        content=resp.body,
        key=link.key,
        bytes=link.bytes,
    )


def tool_result(receipt: RunReceipt, outputs: List[ToolOutput]) -> ToolResult:
    return ToolResult(
        module=receipt.module,
        outputs=tuple(outputs),
        id=receipt.id,
        expires_at=receipt.expires_at,
        cost=receipt.cost,
        credits_balance=receipt.credits_balance,
        message=receipt.message,
    )


def as_timeout(value: Union[float, httpx.Timeout, None]) -> httpx.Timeout:
    if value is None:
        return DEFAULT_TIMEOUT
    if isinstance(value, httpx.Timeout):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return httpx.Timeout(float(value))
    raise ConfigurationError("timeout must be a positive number of seconds, or an httpx.Timeout.")


def with_read_floor(timeout: httpx.Timeout, floor: Optional[float]) -> httpx.Timeout:
    if floor is None or timeout.read is None or timeout.read >= floor:
        return timeout
    return httpx.Timeout(
        connect=timeout.connect, read=floor, write=timeout.write, pool=timeout.pool
    )


def check_job_timeout(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        if math.isfinite(value):
            return float(value)
    raise ConfigurationError("job_timeout must be a positive number of seconds.")


def check_retries(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= MAX_RETRIES_LIMIT:
        raise ConfigurationError(f"max_retries must be an integer from 0 to {MAX_RETRIES_LIMIT}.")
    return value


def check_positive(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer.")
    return value


def connection_error(failure: Failure) -> APIConnectionError:
    hint = (
        " The request may have reached the server: check your credit balance "
        "before retrying a tool run."
        if failure.request_sent
        else ""
    )
    if failure.timed_out:
        return APITimeoutError(
            f"The Pixlub API did not answer in time ({failure.kind}).{hint}",
            request_sent=failure.request_sent,
        )
    return APIConnectionError(
        "Could not reach the Pixlub API "
        f"({failure.kind}). Check the base URL and your network.{hint}",
        request_sent=failure.request_sent,
    )


def _json_object(body: bytes) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(body)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _positive_int(value: Any) -> Optional[int]:
    number = _as_int(value)
    return number if number is not None and number > 0 else None


def _seconds(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        if math.isfinite(value):
            return float(value)
    return None


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _strings(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))

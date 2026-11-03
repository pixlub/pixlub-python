from __future__ import annotations

import asyncio
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pytest

import pixlub
from pixlub import _core

API_KEY = "pix.AbCdEf12.Secret0123456789abcdefghijklmnopqrstuv"
BASE_URL = "https://api.pixlub.test"
FILES_HOST = "files.pixlub.com"

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

RUN_ID = "0b3e5c2a-1f4d-4e8a-9c71-2d6f8a9b0c1e"
JOB_ID = "3c9e5a20-7f41-4b8e-9d2a-6a51f0b4c8e7"
JOB_PATH = f"/v1/jobs/{JOB_ID}"
IMAGE_PATH = f"/r/{RUN_ID}_result.png"
DATA_PATH = f"/r/{RUN_ID}_result.json"
IMAGE_LINK = f"https://{FILES_HOST}{IMAGE_PATH}?exp=1791848556&sig=signature"
DATA_LINK = f"https://{FILES_HOST}{DATA_PATH}?exp=1791848556&sig=signature"


def _input(id: str, required: bool = True, **extra: Any) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "id": id,
        "label": id.title(),
        "description": f"The {id}.",
        "type": "image",
        "required": required,
        "multiple": False,
        "per_file": False,
        "max_files": 1,
        "accept": ["image/png", "image/jpeg", "image/webp"],
    }
    entry.update(extra)
    return entry


IMAGE_OUTPUT = {
    "key": "out",
    "label": "Image",
    "description": "The produced image.",
    "kind": "image",
    "multiple": False,
    "content_types": ["image/png"],
}

CATALOG = [
    {
        "id": "removeBgApi",
        "name": "Remove background",
        "description": "Removes the background.",
        "category": "filter",
        "executable": True,
        "handler": "remote",
        "inputs": [_input("image")],
        "outputs": [IMAGE_OUTPUT],
    },
    {
        "id": "mask",
        "name": "Mask",
        "description": "Applies a mask.",
        "category": "compose",
        "executable": True,
        "handler": "local",
        "inputs": [_input("image"), _input("mask")],
        "outputs": [IMAGE_OUTPUT],
    },
    {
        "id": "checker",
        "name": "Checkerboard",
        "description": "Generates a checkerboard.",
        "category": "generate",
        "executable": True,
        "handler": "local",
        "inputs": [],
        "outputs": [IMAGE_OUTPUT],
    },
    {
        "id": "noise",
        "name": "Noise",
        "description": "Adds noise.",
        "category": "generate",
        "executable": True,
        "handler": "local",
        "inputs": [_input("blend", required=False)],
        "outputs": [IMAGE_OUTPUT],
    },
    {
        "id": "collage",
        "name": "Collage",
        "description": "Arranges several images on a background.",
        "category": "compose",
        "executable": True,
        "handler": "local",
        "inputs": [
            _input("background"),
            _input("tiles", multiple=True, max_files=4),
        ],
        "outputs": [IMAGE_OUTPUT],
    },
    {
        "id": "flipBatch",
        "name": "Flip (batch)",
        "description": "Each image is a run.",
        "category": "transform",
        "executable": True,
        "handler": "local",
        "inputs": [_input("image", multiple=True, per_file=True, max_files=10)],
        "outputs": [{**IMAGE_OUTPUT, "multiple": True}],
    },
    {
        "id": "stack",
        "name": "Stack",
        "description": "Stacks images, with no file limit of its own.",
        "category": "compose",
        "executable": True,
        "handler": "local",
        "inputs": [_input("files", multiple=True, max_files=None)],
        "outputs": [IMAGE_OUTPUT],
    },
    {
        "id": "trimVideo",
        "name": "Trim a video",
        "description": "Video only.",
        "category": "transform",
        "executable": False,
        "handler": "none",
        "inputs": [],
        "outputs": [],
    },
]


def json_response(
    payload: Any, status: int = 200, headers: Optional[Dict[str, str]] = None
) -> httpx.Response:
    return httpx.Response(status, json=payload, headers=headers)


def image_response(data: bytes = PNG, headers: Optional[Dict[str, str]] = None) -> httpx.Response:
    return httpx.Response(
        200, content=data, headers={"content-type": "image/png", **(headers or {})}
    )


def run_payload(outputs: Optional[List[Dict[str, Any]]] = None, **extra: Any) -> Dict[str, Any]:
    if outputs is None:
        outputs = [
            {
                "key": "out",
                "name": "image",
                "url": IMAGE_LINK,
                "content_type": "image/png",
                "bytes": len(PNG),
            }
        ]
    payload: Dict[str, Any] = {
        "id": RUN_ID,
        "module": "removeBgApi",
        "outputs": outputs,
        "expires_at": "2026-10-13T10:00:00Z",
        "credits": {"cost": 10, "balance": 41},
    }
    payload.update(extra)
    return payload


def run_response(outputs: Optional[List[Dict[str, Any]]] = None, **extra: Any) -> httpx.Response:
    return json_response(run_payload(outputs, **extra))


def accepted(job_id: str = JOB_ID, module: str = "removeBgApi") -> httpx.Response:
    return json_response(
        {
            "job_id": job_id,
            "module": module,
            "status": "running",
            "poll": f"https://api.pixlub.com/v1/jobs/{job_id}",
        },
        status=202,
        headers={"retry-after": "2"},
    )


def job(status: str, **extra: Any) -> httpx.Response:
    payload: Dict[str, Any] = {
        "job_id": JOB_ID,
        "module": "removeBgApi",
        "status": status,
        "created_at": "2026-10-13T09:00:00Z",
    }
    if status in ("succeeded", "failed"):
        payload["finished_at"] = "2026-10-13T09:01:30Z"
    payload.update(extra)
    return json_response(payload)


def succeeded(**extra: Any) -> httpx.Response:
    payload = run_payload(**extra)
    payload.update(
        {
            "job_id": JOB_ID,
            "status": "succeeded",
            "created_at": "2026-10-13T09:00:00Z",
            "finished_at": "2026-10-13T09:01:30Z",
        }
    )
    return json_response(payload)


class FakeApi:
    def __init__(self) -> None:
        self.requests: List[httpx.Request] = []
        self.uploaded: Dict[str, bytes] = {}
        self.upload_ids: List[str] = []
        self._queued: Dict[Tuple[str, str], List[Any]] = {}

    def queue(self, method: str, path: str, *answers: Any) -> None:
        self._queued.setdefault((method, path), []).extend(answers)

    def calls(self, method: str, path: str) -> List[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path == path]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        queue = self._queued.get((request.method, request.url.path))
        if queue:
            answer = queue.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            if callable(answer):
                return answer(request)
            return answer
        return self._default(request)

    def _default(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == "/v1/modules":
            return json_response(CATALOG)
        if request.method == "GET" and path.endswith("/schema"):
            return json_response(
                {"type": "object", "additionalProperties": False, "properties": {}}
            )
        if request.method == "POST" and path == "/v1/uploads":
            upload_id = str(uuid.uuid4())
            self.upload_ids.append(upload_id)
            return json_response(
                {
                    "upload_id": upload_id,
                    "upload_url": f"https://{FILES_HOST}/u/{upload_id}?exp=1791848556&sig=s",
                    "method": "PUT",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
                status=201,
            )
        if request.method == "PUT" and path.startswith("/u/"):
            body = request.read()
            self.uploaded[path[len("/u/") :]] = body
            return json_response({"bytes": len(body)})
        if request.method == "POST" and path.startswith("/v1/modules/"):
            return run_response()
        if request.method == "GET" and path.endswith("_result.png"):
            return image_response()
        if request.method == "GET" and path.endswith("_result.json"):
            return json_response({"faces": []})
        return json_response({"error": "Unknown route"}, status=404)


def parse_multipart(request: httpx.Request) -> Dict[str, Dict[str, Any]]:
    return {name: part for name, part in parse_multipart_all(request)}


def parse_multipart_all(request: httpx.Request) -> List[Tuple[str, Dict[str, Any]]]:
    content_type = request.headers["content-type"]
    boundary = content_type.split("boundary=", 1)[1].strip('"').encode()
    body = request.read()
    assert int(request.headers["content-length"]) == len(body)
    assert "transfer-encoding" not in request.headers
    parts: List[Tuple[str, Dict[str, Any]]] = []
    for chunk in body.split(b"--" + boundary):
        if chunk in (b"", b"--", b"--\r\n"):
            continue
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        if chunk.endswith(b"\r\n"):
            chunk = chunk[:-2]
        head, _, content = chunk.partition(b"\r\n\r\n")
        headers = head.decode()
        name = re.search(r'name="([^"]*)"', headers)
        filename = re.search(r'filename="([^"]*)"', headers)
        ctype = re.search(r"Content-Type: (\S+)", headers)
        assert name is not None
        parts.append(
            (
                name.group(1),
                {
                    "filename": filename.group(1) if filename else None,
                    "content_type": ctype.group(1) if ctype else None,
                    "content": content,
                },
            )
        )
    return parts


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(_core, "clock", fake)
    return fake


@pytest.fixture(autouse=True)
def waits(monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> List[float]:
    recorded: List[float] = []

    def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)
        clock.now += seconds

    async def fake_async_sleep(seconds: float, *args: Any, **kwargs: Any) -> None:
        recorded.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(time, "sleep", fake_sleep)
    monkeypatch.setattr(asyncio, "sleep", fake_async_sleep)
    return recorded


@pytest.fixture
def api() -> FakeApi:
    return FakeApi()


@pytest.fixture
def make_client(api: FakeApi):
    clients: List[pixlub.Client] = []

    def factory(**kwargs: Any) -> pixlub.Client:
        kwargs.setdefault("http_client", httpx.Client(transport=httpx.MockTransport(api)))
        client = pixlub.Client(API_KEY, base_url=BASE_URL, **kwargs)
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.close()


@pytest.fixture
def make_async_client(api: FakeApi):

    def factory(**kwargs: Any) -> pixlub.AsyncClient:
        kwargs.setdefault("http_client", httpx.AsyncClient(transport=httpx.MockTransport(api)))
        return pixlub.AsyncClient(API_KEY, base_url=BASE_URL, **kwargs)

    return factory


@pytest.fixture
def client(make_client) -> pixlub.Client:
    return make_client()


@pytest.fixture
def small_direct_limit(monkeypatch: pytest.MonkeyPatch) -> int:
    limit = 100
    monkeypatch.setattr(_core, "DIRECT_UPLOAD_MAX_BYTES", limit)
    return limit

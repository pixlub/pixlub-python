from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import httpx
import pytest
from conftest import (
    API_KEY,
    FILES_HOST,
    PNG,
    json_response,
    parse_multipart,
    parse_multipart_all,
)

import pixlub
from pixlub.errors import (
    APITimeoutError,
    FileTooLargeError,
    InvalidInputError,
    InvalidRequestError,
    NotFoundError,
    ServerError,
    UnexpectedResponseError,
)

BIG = PNG + b"\x00" * 500  # over the shrunk direct limit
UPLOAD_A = "6f1c2d3e-4a5b-4c6d-8e7f-001122334455"
UPLOAD_B = "7a2b3c4d-5e6f-4a1b-9c2d-66778899aabb"
RUN = "/v1/modules/removeBgApi"


@pytest.fixture(params=["sync", "async"])
def run_tool(request, make_client, make_async_client) -> Callable[..., Any]:

    def run(tool_id: str, client_options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        options = client_options or {}
        if request.param == "sync":
            return make_client(**options).tools.run(tool_id, **kwargs)

        async def scenario() -> Any:
            async with make_async_client(**options) as client:
                return await client.tools.run(tool_id, **kwargs)

        return asyncio.run(scenario())

    return run


def ticket(upload_id: str, *, expires_at: str = "2099-01-01T00:00:00Z", url: str = "") -> Any:
    return json_response(
        {
            "upload_id": upload_id,
            "upload_url": url or f"https://{FILES_HOST}/u/{upload_id}?exp=1&sig=s",
            "method": "PUT",
            "expires_at": expires_at,
        },
        status=201,
    )


def error(status: int, code: str, **extra: Any) -> Any:
    return json_response({"error": code.replace("_", " "), "code": code, **extra}, status=status)


def big_file(tmp_path: Path, name: str = "big.png", data: bytes = BIG) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_small_files_are_sent_directly(api, run_tool, small_direct_limit) -> None:
    run_tool("removeBgApi", image=PNG)

    assert api.calls("POST", "/v1/uploads") == []
    fields = parse_multipart(api.calls("POST", RUN)[0])
    assert fields["image"]["content"] == PNG
    assert "upload_id" not in fields


def test_a_large_file_is_uploaded_then_run_by_its_id(
    api, run_tool, small_direct_limit, tmp_path: Path
) -> None:
    result = run_tool("removeBgApi", image=big_file(tmp_path))

    assert result.image == PNG
    opened = api.calls("POST", "/v1/uploads")
    assert len(opened) == 1
    assert opened[0].headers["X-API-Key"] == API_KEY
    assert json.loads(opened[0].read()) == {
        "module": "removeBgApi",
        "filename": "big.png",
        "bytes": len(BIG),
    }

    upload_id = api.upload_ids[0]
    put = api.calls("PUT", f"/u/{upload_id}")[0]
    assert put.url.host == FILES_HOST
    assert "x-api-key" not in put.headers
    assert API_KEY not in str(put.url)
    assert put.headers["content-length"] == str(len(BIG))
    assert "transfer-encoding" not in put.headers
    assert put.headers["content-type"] == "image/png"
    assert api.uploaded[upload_id] == BIG

    fields = parse_multipart(api.calls("POST", RUN)[0])
    assert set(fields) == {"params", "upload_id"}
    assert fields["upload_id"]["content"] == upload_id.encode()
    order = [(r.method, r.url.path) for r in api.requests if r.url.path != "/v1/modules"]
    assert order[:3] == [("POST", "/v1/uploads"), ("PUT", f"/u/{upload_id}"), ("POST", RUN)]


def test_the_other_inputs_stay_in_the_run_request(
    api, run_tool, small_direct_limit, tmp_path: Path
) -> None:
    run_tool("mask", inputs={"image": big_file(tmp_path), "mask": PNG})

    fields = parse_multipart(api.calls("POST", "/v1/modules/mask")[0])
    assert set(fields) == {"params", "upload_id", "mask"}
    assert fields["mask"]["content"] == PNG
    assert list(api.uploaded.values()) == [BIG]


class OneWayStream(io.RawIOBase):
    def __init__(self, data: bytes) -> None:
        self._data = io.BytesIO(data)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, buffer) -> int:  # type: ignore[no-untyped-def]
        chunk = self._data.read(len(buffer))
        buffer[: len(chunk)] = chunk
        return len(chunk)


def test_a_stream_that_cannot_seek_is_uploaded_whole(api, run_tool, small_direct_limit) -> None:
    run_tool("removeBgApi", image=OneWayStream(BIG))

    assert list(api.uploaded.values()) == [BIG]


def test_a_seekable_stream_is_uploaded_from_its_position(api, run_tool, small_direct_limit) -> None:
    stream = io.BytesIO(b"skip me" + BIG)
    stream.seek(7)
    run_tool("removeBgApi", image=stream)

    assert list(api.uploaded.values()) == [BIG]


@pytest.mark.parametrize(
    ("tool_id", "inputs", "message"),
    [
        ("flipBatch", {"image": [BIG, BIG]}, "split them"),
        ("mask", {"image": PNG, "mask": BIG}, "other inputs"),
    ],
)
def test_runs_that_fit_no_way_are_refused_locally(
    api, run_tool, small_direct_limit, tool_id: str, inputs: Dict[str, Any], message: str
) -> None:
    with pytest.raises(InvalidInputError, match=message):
        run_tool(tool_id, inputs=inputs)
    assert [r for r in api.requests if r.method in ("POST", "PUT")] == []


def test_put_is_retried_on_server_and_network_errors(
    api, run_tool, small_direct_limit, tmp_path: Path
) -> None:
    api.queue("POST", "/v1/uploads", ticket(UPLOAD_A))
    api.queue(
        "PUT",
        f"/u/{UPLOAD_A}",
        error(507, "insufficient_storage"),
        httpx.ReadTimeout("slow"),
    )

    run_tool("removeBgApi", image=big_file(tmp_path))

    puts = api.calls("PUT", f"/u/{UPLOAD_A}")
    assert len(puts) == 3
    assert all(p.read() == BIG for p in puts)
    assert len(api.calls("POST", "/v1/uploads")) == 1
    assert len(api.calls("POST", RUN)) == 1


def test_put_gives_up_after_the_retries(api, run_tool, small_direct_limit, tmp_path: Path) -> None:
    api.queue("POST", "/v1/uploads", ticket(UPLOAD_A))
    api.queue("PUT", f"/u/{UPLOAD_A}", *[error(500, "upload_failed") for _ in range(3)])

    with pytest.raises(ServerError):
        run_tool("removeBgApi", {"max_retries": 2}, image=big_file(tmp_path))

    assert len(api.calls("PUT", f"/u/{UPLOAD_A}")) == 3
    assert api.calls("POST", RUN) == []


def test_put_too_large_is_not_retried(api, run_tool, small_direct_limit, tmp_path: Path) -> None:
    api.queue("POST", "/v1/uploads", ticket(UPLOAD_A))
    api.queue("PUT", f"/u/{UPLOAD_A}", error(413, "file_too_large", max_bytes=400))

    with pytest.raises(FileTooLargeError) as caught:
        run_tool("removeBgApi", image=big_file(tmp_path))

    assert caught.value.max_bytes == 400
    assert len(api.calls("PUT", f"/u/{UPLOAD_A}")) == 1


def test_an_expired_link_opens_one_new_upload(
    api, run_tool, small_direct_limit, tmp_path: Path
) -> None:
    api.queue("POST", "/v1/uploads", ticket(UPLOAD_A), ticket(UPLOAD_B))
    api.queue("PUT", f"/u/{UPLOAD_A}", error(403, "invalid_link"))

    run_tool("removeBgApi", image=big_file(tmp_path))

    assert len(api.calls("POST", "/v1/uploads")) == 2
    assert api.uploaded == {UPLOAD_B: BIG}
    fields = parse_multipart(api.calls("POST", RUN)[0])
    assert fields["upload_id"]["content"] == UPLOAD_B.encode()


def test_two_expired_links_in_a_row_stop(api, run_tool, small_direct_limit, tmp_path: Path) -> None:
    api.queue("POST", "/v1/uploads", ticket(UPLOAD_A), ticket(UPLOAD_B))
    api.queue("PUT", f"/u/{UPLOAD_A}", error(403, "invalid_link"))
    api.queue("PUT", f"/u/{UPLOAD_B}", error(403, "invalid_link"))

    with pytest.raises(UnexpectedResponseError, match="expired"):
        run_tool("removeBgApi", image=big_file(tmp_path))

    assert len(api.calls("POST", "/v1/uploads")) == 2
    assert api.calls("POST", RUN) == []


def test_a_link_past_its_expiry_is_not_retried(
    api, run_tool, small_direct_limit, tmp_path: Path
) -> None:
    api.queue("POST", "/v1/uploads", ticket(UPLOAD_A, expires_at="2001-01-01T00:00:00Z"))
    api.queue("PUT", f"/u/{UPLOAD_A}", error(503, "unavailable"))

    run_tool("removeBgApi", image=big_file(tmp_path))

    assert len(api.calls("PUT", f"/u/{UPLOAD_A}")) == 1
    assert len(api.calls("POST", "/v1/uploads")) == 2


def test_put_follows_no_redirect(api, run_tool, small_direct_limit, tmp_path: Path) -> None:
    api.queue("POST", "/v1/uploads", ticket(UPLOAD_A))
    api.queue(
        "PUT",
        f"/u/{UPLOAD_A}",
        httpx.Response(307, headers={"location": "https://attacker.test/steal"}),
    )

    with pytest.raises(pixlub.APIError) as caught:
        run_tool("removeBgApi", image=big_file(tmp_path))

    assert caught.value.http_status == 307
    assert "attacker.test" not in {r.url.host for r in api.requests}


@pytest.mark.parametrize(
    "link",
    [
        "http://files.pixlub.com/u/x",  # plain http outside localhost
        "https://user:pw@files.pixlub.com/u/x",
        "ftp://files.pixlub.com/u/x",
    ],
)
def test_an_unsafe_upload_link_is_never_used(
    api, run_tool, small_direct_limit, tmp_path: Path, link: str
) -> None:
    api.queue("POST", "/v1/uploads", ticket(UPLOAD_A, url=link))

    with pytest.raises(UnexpectedResponseError, match="Refused an upload link"):
        run_tool("removeBgApi", image=big_file(tmp_path))

    assert [r for r in api.requests if r.method == "PUT"] == []


def test_opening_an_upload_is_retried_on_429_only(
    api, run_tool, small_direct_limit, tmp_path: Path, waits
) -> None:
    api.queue(
        "POST",
        "/v1/uploads",
        json_response({"error": "slow"}, 429, {"retry-after": "3"}),
        error(503, "storage_unavailable"),
    )

    with pytest.raises(pixlub.ServiceUnavailableError):
        run_tool("removeBgApi", image=big_file(tmp_path))

    assert len(api.calls("POST", "/v1/uploads")) == 2
    assert waits == [3.0]


def test_a_size_refused_on_opening_exposes_the_limit(
    api, run_tool, small_direct_limit, tmp_path: Path
) -> None:
    api.queue("POST", "/v1/uploads", error(413, "file_too_large", max_bytes=10485760))

    with pytest.raises(FileTooLargeError) as caught:
        run_tool("removeBgApi", image=big_file(tmp_path))

    assert caught.value.code == "file_too_large"
    assert caught.value.max_bytes == 10485760
    assert [r for r in api.requests if r.method == "PUT"] == []


def queued(upload_id: str, position: int) -> Any:
    return json_response(
        {"upload_id": upload_id, "status": "queued", "position": position}, status=202
    )


def test_a_queued_upload_is_followed_until_its_link(
    api, run_tool, small_direct_limit, tmp_path: Path, waits
) -> None:
    api.queue("POST", "/v1/uploads", queued(UPLOAD_A, 2))
    api.queue("GET", f"/v1/uploads/{UPLOAD_A}", queued(UPLOAD_A, 1), ticket(UPLOAD_A))

    result = run_tool("removeBgApi", image=big_file(tmp_path))

    assert result.image == PNG
    follows = api.calls("GET", f"/v1/uploads/{UPLOAD_A}")
    assert len(follows) == 2
    for follow in follows:
        assert follow.headers["X-API-Key"] == API_KEY
        assert follow.headers["Prefer"] == "wait=25"
    assert waits == [2.0]
    assert api.uploaded[UPLOAD_A] == BIG
    fields = parse_multipart(api.calls("POST", RUN)[0])
    assert fields["upload_id"]["content"] == UPLOAD_A.encode()


def test_a_queued_upload_that_is_given_up_fails(
    api, run_tool, small_direct_limit, tmp_path: Path
) -> None:
    api.queue("POST", "/v1/uploads", queued(UPLOAD_A, 1))
    api.queue("GET", f"/v1/uploads/{UPLOAD_A}", error(404, "upload_not_found"))

    with pytest.raises(NotFoundError):
        run_tool("removeBgApi", image=big_file(tmp_path))

    assert [r for r in api.requests if r.method == "PUT"] == []
    assert api.calls("POST", RUN) == []


def test_a_queue_answer_for_another_upload_is_refused(
    api, run_tool, small_direct_limit, tmp_path: Path
) -> None:
    api.queue("POST", "/v1/uploads", queued(UPLOAD_A, 1))
    api.queue("GET", f"/v1/uploads/{UPLOAD_A}", ticket(UPLOAD_B))

    with pytest.raises(UnexpectedResponseError, match="another upload"):
        run_tool("removeBgApi", image=big_file(tmp_path))

    assert [r for r in api.requests if r.method == "PUT"] == []


def test_upload_incomplete_is_retried_with_growing_pauses(
    api, run_tool, small_direct_limit, tmp_path: Path, waits
) -> None:
    api.queue("POST", RUN, error(409, "upload_incomplete"), error(409, "upload_incomplete"))

    result = run_tool("removeBgApi", {"max_retries": 0}, image=big_file(tmp_path))

    assert result.image == PNG
    runs = api.calls("POST", RUN)
    assert len(runs) == 3
    ids = {parse_multipart(r)["upload_id"]["content"] for r in runs}
    assert ids == {api.upload_ids[0].encode()}
    assert waits == [1.0, 2.0]


def test_upload_incomplete_gives_up_after_five_tries(
    api, run_tool, small_direct_limit, tmp_path: Path, waits
) -> None:
    api.queue("POST", RUN, *[error(409, "upload_incomplete") for _ in range(6)])

    with pytest.raises(InvalidRequestError) as caught:
        run_tool("removeBgApi", image=big_file(tmp_path))

    assert caught.value.code == "upload_incomplete"
    assert len(api.calls("POST", RUN)) == 5
    assert waits == [1.0, 2.0, 4.0, 8.0]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (lambda: error(500, "boom"), ServerError),
        (lambda: error(404, "upload_not_found"), NotFoundError),
        (lambda: httpx.ReadTimeout("slow"), APITimeoutError),
        (lambda: error(409, "something_else"), InvalidRequestError),
    ],
)
def test_an_upload_id_is_never_sent_twice_after_other_answers(
    api, run_tool, small_direct_limit, tmp_path: Path, answer: Any, expected: type
) -> None:
    api.queue("POST", RUN, answer())

    with pytest.raises(expected):
        run_tool("removeBgApi", image=big_file(tmp_path))

    assert len(api.calls("POST", RUN)) == 1
    assert len(api.calls("POST", "/v1/uploads")) == 1


def test_a_429_on_the_run_keeps_the_upload_id(
    api, run_tool, small_direct_limit, tmp_path: Path
) -> None:
    api.queue("POST", RUN, json_response({"error": "slow"}, 429, {"retry-after": "1"}))

    run_tool("removeBgApi", image=big_file(tmp_path))

    runs = api.calls("POST", RUN)
    assert len(runs) == 2
    assert parse_multipart(runs[0])["upload_id"] == parse_multipart(runs[1])["upload_id"]


def test_a_413_use_uploads_switches_to_an_upload_once(api, run_tool) -> None:
    api.queue("POST", RUN, error(413, "use_uploads", max_bytes=94371840))

    result = run_tool("removeBgApi", image=PNG)

    assert result.image == PNG
    first, second = api.calls("POST", RUN)
    assert "image" in parse_multipart(first)
    assert set(parse_multipart(second)) == {"params", "upload_id"}
    assert len(api.calls("POST", "/v1/uploads")) == 1
    assert list(api.uploaded.values()) == [PNG]


def test_a_second_413_is_raised(api, run_tool) -> None:
    api.queue(
        "POST",
        "/v1/modules/mask",
        error(413, "use_uploads", max_bytes=10),
        error(413, "use_uploads", max_bytes=10),
    )

    with pytest.raises(FileTooLargeError) as caught:
        run_tool("mask", inputs={"image": PNG, "mask": PNG[:8]})

    assert caught.value.code == "use_uploads"
    assert len(api.calls("POST", "/v1/modules/mask")) == 2
    assert len(api.calls("POST", "/v1/uploads")) == 1


def test_a_413_with_no_way_to_upload_is_raised(api, run_tool) -> None:
    api.queue("POST", "/v1/modules/flipBatch", error(413, "use_uploads", max_bytes=10))

    with pytest.raises(FileTooLargeError):
        run_tool("flipBatch", inputs={"image": [PNG, PNG]})

    assert api.calls("POST", "/v1/uploads") == []


def test_a_source_url_is_sent_as_a_field(api, run_tool) -> None:
    run_tool("removeBgApi", image=pixlub.SourceURL("https://example.com/photo.jpg"))

    fields = parse_multipart(api.calls("POST", RUN)[0])
    assert set(fields) == {"params", "source_url"}
    assert fields["source_url"]["content"] == b"https://example.com/photo.jpg"
    assert fields["source_url"]["filename"] is None
    assert {r.url.host for r in api.requests} == {"api.pixlub.test", FILES_HOST}
    assert api.calls("POST", "/v1/uploads") == []


def test_a_source_url_keeps_the_other_inputs(api, run_tool) -> None:
    run_tool("mask", inputs={"image": pixlub.SourceURL("https://example.com/a.png"), "mask": PNG})

    fields = parse_multipart(api.calls("POST", "/v1/modules/mask")[0])
    assert set(fields) == {"params", "source_url", "mask"}


@pytest.mark.parametrize(
    "inputs",
    [
        {"image": PNG, "mask": pixlub.SourceURL("https://example.com/m.png")},  # not the main input
        {
            "image": [
                pixlub.SourceURL("https://example.com/a.png"),
                pixlub.SourceURL("https://example.com/b.png"),
            ],
            "mask": PNG,
        },
    ],
)
def test_misplaced_source_urls_are_refused(api, run_tool, inputs: Dict[str, Any]) -> None:
    with pytest.raises(InvalidInputError, match="SourceURL"):
        run_tool("mask", inputs=inputs)
    assert api.calls("POST", "/v1/modules/mask") == []


def test_a_plain_string_is_still_a_path(api, run_tool) -> None:
    with pytest.raises(InvalidInputError, match="Cannot read"):
        run_tool("removeBgApi", image="https://example.com/photo.jpg")
    assert api.calls("POST", RUN) == []


def test_an_input_that_takes_several_files_gets_a_list(api, run_tool) -> None:
    tiles = [PNG, PNG + b"1", PNG + b"2"]
    run_tool("collage", inputs={"background": PNG, "tiles": tiles})

    parts = parse_multipart_all(api.calls("POST", "/v1/modules/collage")[0])
    assert [name for name, _ in parts] == ["params", "background", "tiles", "tiles", "tiles"]
    assert [p["content"] for name, p in parts if name == "tiles"] == tiles


def test_a_batch_input_takes_a_list_through_image(api, run_tool) -> None:
    run_tool("flipBatch", image=[PNG, PNG])

    parts = parse_multipart_all(api.calls("POST", "/v1/modules/flipBatch")[0])
    assert [name for name, _ in parts] == ["params", "image", "image"]


@pytest.mark.parametrize(
    ("tool_id", "inputs", "message"),
    [
        ("collage", {"background": PNG, "tiles": [PNG] * 5}, "at most 4 files"),
        ("collage", {"background": [PNG, PNG], "tiles": [PNG]}, "takes one file"),
        ("collage", {"background": PNG, "tiles": []}, "empty list"),
        ("stack", {"files": [PNG] * 65}, "at most 64 files"),
    ],
)
def test_file_counts_are_checked_locally(
    api, run_tool, tool_id: str, inputs: Dict[str, Any], message: str
) -> None:
    with pytest.raises(InvalidInputError, match=message):
        run_tool(tool_id, inputs=inputs)
    assert [r for r in api.requests if r.method == "POST"] == []


def test_sixty_four_files_are_accepted(api, run_tool) -> None:
    run_tool("stack", inputs={"files": [PNG] * 64})
    parts = parse_multipart_all(api.calls("POST", "/v1/modules/stack")[0])
    assert len(parts) == 65

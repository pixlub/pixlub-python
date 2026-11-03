from __future__ import annotations

import asyncio
import email.utils
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from conftest import PNG

from pixlub import _core
from pixlub._files import CHUNK_SIZE, MultipartBody, SourceURL, prepare_file, sniff_content_type
from pixlub._retry import (
    MAX_RETRY_AFTER,
    backoff_seconds,
    failure_for,
    parse_retry_after,
    status_is_retryable,
    upload_incomplete_pause,
)
from pixlub.errors import (
    FileTooLargeError,
    InsufficientCreditsError,
    InvalidInputError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    UnexpectedResponseError,
)
from pixlub.models import ToolOutput, ToolResult

LIMIT = 1024


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (PNG, "image/png"),
        (b"\xff\xd8\xff\xe0" + b"0" * 8, "image/jpeg"),
        (b"GIF89a" + b"0" * 8, "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"BM" + b"0" * 14, "image/bmp"),
        (b"II*\x00" + b"0" * 12, "image/tiff"),
        (b"\x00\x00\x00\x18ftypavif" + b"0" * 4, "image/avif"),
        (b"just some text", None),
    ],
)
def test_sniff_recognises_common_formats(data: bytes, expected: str | None) -> None:
    assert sniff_content_type(data) == expected


def _content(file) -> bytes:
    return b"".join(file.chunks())


def test_bytes_input_gets_a_filename_from_its_type() -> None:
    upload = prepare_file(PNG, form_name="image", max_bytes=LIMIT)
    assert upload.filename == "image.png"
    assert upload.content_type == "image/png"
    assert upload.size == len(PNG)
    assert _content(upload) == PNG


def test_unknown_content_is_sent_as_octet_stream() -> None:
    upload = prepare_file(b"not an image", form_name="image", max_bytes=LIMIT)
    assert upload.content_type == "application/octet-stream"
    assert upload.filename == "image.img"


def test_filename_from_a_path_is_sanitised() -> None:
    stream = io.BytesIO(PNG)
    stream.name = '/tmp/evil"name\r\nX-Injected: 1.png'  # type: ignore[attr-defined]
    upload = prepare_file(stream, form_name="image", max_bytes=LIMIT)
    assert '"' not in upload.filename
    assert "\r" not in upload.filename and "\n" not in upload.filename


def test_empty_input_is_refused() -> None:
    with pytest.raises(InvalidInputError, match="empty"):
        prepare_file(b"", form_name="image", max_bytes=LIMIT)


def test_oversized_bytes_are_refused() -> None:
    with pytest.raises(InvalidInputError, match="limit is 1024 bytes"):
        prepare_file(PNG * 100, form_name="image", max_bytes=LIMIT)


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

    def consumed(self) -> int:
        return self._data.tell()


def test_seekable_stream_is_measured_without_being_read() -> None:
    stream = io.BytesIO(b"x" * (LIMIT * 10))
    with pytest.raises(InvalidInputError, match="limit is 1024 bytes"):
        prepare_file(stream, form_name="image", max_bytes=LIMIT)
    assert stream.tell() == 0


def test_seekable_stream_is_sent_from_its_current_position() -> None:
    stream = io.BytesIO(b"header" + PNG)
    stream.seek(6)
    upload = prepare_file(stream, form_name="image", max_bytes=LIMIT)
    assert upload.size == len(PNG)
    assert upload.content_type == "image/png"
    assert _content(upload) == PNG
    assert _content(upload) == PNG


def test_one_way_stream_is_copied_only_up_to_the_limit() -> None:
    stream = OneWayStream(b"x" * (LIMIT * 10))
    with pytest.raises(InvalidInputError):
        prepare_file(stream, form_name="image", max_bytes=LIMIT)  # type: ignore[arg-type]
    assert stream.consumed() == LIMIT + 1


def test_one_way_stream_is_spooled_and_can_be_read_again() -> None:
    upload = prepare_file(OneWayStream(PNG), form_name="image", max_bytes=LIMIT)  # type: ignore[arg-type]
    assert upload.size == len(PNG)
    assert _content(upload) == PNG
    assert _content(upload) == PNG
    upload.close()


def test_text_mode_file_object_is_refused() -> None:
    with pytest.raises(TypeError, match="binary mode"):
        prepare_file(io.StringIO("not bytes"), form_name="image", max_bytes=LIMIT)  # type: ignore[arg-type]


def test_path_to_a_real_file_is_measured_then_streamed(tmp_path: Path) -> None:
    source = tmp_path / "scan.webp"
    data = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"0" * (CHUNK_SIZE * 2 + 5)
    source.write_bytes(data)
    upload = prepare_file(source, form_name="image", max_bytes=len(data))
    assert upload.filename == "scan.webp"
    assert upload.content_type == "image/webp"
    assert upload.size == len(data)
    chunks = list(upload.chunks())
    assert len(chunks) == 3 and max(len(c) for c in chunks) <= CHUNK_SIZE
    assert b"".join(chunks) == data


def test_a_file_that_shrinks_while_sent_is_an_error(tmp_path: Path) -> None:
    source = tmp_path / "in.png"
    source.write_bytes(PNG)
    upload = prepare_file(source, form_name="image", max_bytes=LIMIT)
    source.write_bytes(PNG[:10])
    with pytest.raises(InvalidInputError, match="got shorter"):
        _content(upload)


def test_multipart_length_is_exact_and_files_are_streamed(tmp_path: Path) -> None:
    source = tmp_path / "写真.png"
    source.write_bytes(PNG)
    files = [
        prepare_file(source, form_name="image", max_bytes=LIMIT),
        prepare_file(b"%PDF-1.7", form_name="doc", max_bytes=LIMIT),
    ]
    body = MultipartBody([("params", "{}"), ("upload_id", "abc")], files)
    content = b"".join(body.chunks())
    assert body.length == len(content)
    assert body.content_type.startswith("multipart/form-data; boundary=")
    assert 'filename="写真.png"'.encode() in content
    assert content.count(PNG) == 1


def test_async_chunks_match_sync_chunks(tmp_path: Path) -> None:
    source = tmp_path / "in.png"
    source.write_bytes(PNG * 10_000)
    upload = prepare_file(source, form_name="image", max_bytes=len(PNG) * 10_000)

    async def collect() -> bytes:
        return b"".join([chunk async for chunk in upload.achunks()])

    assert asyncio.run(collect()) == _content(upload)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/a.png",  # not https
        "https://example.com:8443/a.png",  # not port 443
        "https://user:pw@example.com/a.png",  # credentials
        "https://",  # no host
        "https://example.com/a b.png",  # whitespace
        "https://example.com/" + "a" * 2048,  # too long
        "",
    ],
)
def test_unsafe_source_urls_are_refused(url: str) -> None:
    with pytest.raises(InvalidInputError):
        SourceURL(url)


def test_source_url_accepts_a_public_https_address() -> None:
    assert (
        SourceURL(" https://example.com:443/a.png?x=1 ").url == "https://example.com:443/a.png?x=1"
    )


@pytest.mark.parametrize(
    ("method", "status", "retry"),
    [
        ("GET", 500, True),
        ("GET", 503, True),
        ("GET", 429, True),
        ("GET", 400, False),
        ("GET", 404, False),
        ("POST", 429, True),  # rate limiter answers before any charge
        ("POST", 500, False),  # the run may already have been charged
        ("POST", 503, False),
        ("POST", 400, False),
        ("PUT", 500, True),
        ("PUT", 507, True),
        ("PUT", 503, True),
        ("PUT", 429, True),
        ("PUT", 403, False),
        ("PUT", 413, False),
    ],
)
def test_status_retry_policy(method: str, status: int, retry: bool) -> None:
    assert status_is_retryable(method, status) is retry


def test_connection_refused_is_retried_for_every_method() -> None:
    for method in ("GET", "POST"):
        failure = failure_for(httpx.ConnectError("refused"), method)
        assert failure.retryable is True
        assert failure.request_sent is False


def test_read_timeout_is_retried_only_for_get() -> None:
    get = failure_for(httpx.ReadTimeout("slow"), "GET")
    post = failure_for(httpx.ReadTimeout("slow"), "POST")
    assert get.retryable is True
    assert post.retryable is False
    assert post.request_sent is True and post.timed_out is True


def test_put_is_retried_after_any_network_error() -> None:
    failure = failure_for(httpx.ReadTimeout("slow"), "PUT")
    assert failure.retryable is True


def test_upload_incomplete_pauses_grow() -> None:
    assert [upload_incomplete_pause(i) for i in range(4)] == [1.0, 2.0, 4.0, 8.0]
    assert upload_incomplete_pause(0, retry_after=3.0) == 3.0


def test_connect_timeout_is_not_sent_and_is_retryable_for_post() -> None:
    failure = failure_for(httpx.ConnectTimeout("slow connect"), "POST")
    assert failure.retryable is True
    assert failure.request_sent is False
    assert failure.timed_out is True


def test_backoff_grows_and_is_capped() -> None:
    assert 0.25 <= backoff_seconds(0) <= 0.5
    assert backoff_seconds(30) <= 8.0


def test_retry_after_seconds_and_cap() -> None:
    assert backoff_seconds(0, retry_after=2.0) == 2.0
    assert backoff_seconds(0, retry_after=10_000.0) == MAX_RETRY_AFTER
    assert backoff_seconds(0, retry_after=-5.0) == 0.0


def test_parse_retry_after_accepts_seconds_and_dates() -> None:
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon, maybe") is None

    in_a_minute = datetime.now(timezone.utc) + timedelta(seconds=60)
    parsed = parse_retry_after(email.utils.format_datetime(in_a_minute))
    assert parsed is not None and 50 <= parsed <= 61


def test_parse_retry_after_in_the_past_is_zero() -> None:
    past = datetime.now(timezone.utc) - timedelta(minutes=5)
    assert parse_retry_after(email.utils.format_datetime(past)) == 0.0


def _image_result(content: bytes = PNG, content_type: str = "image/png") -> ToolResult:
    output = ToolOutput(name="image", url="https://x/r", content_type=content_type, content=content)
    return ToolResult(module="flip", outputs=(output,))


def test_tool_result_repr_does_not_dump_the_image() -> None:
    result = _image_result()
    assert "\\x89PNG" not in repr(result)
    assert result.extension == ".png"


def test_tool_result_save_uses_the_given_path(tmp_path: Path) -> None:
    result = _image_result()
    target = result.save(str(tmp_path / "a.png"))
    assert isinstance(target, Path)
    assert target.read_bytes() == PNG


def test_unknown_content_type_extension_is_generic() -> None:
    assert _image_result(b"x", "image/x-weird").extension == ".img"


def test_pdf_output_counts_as_the_produced_file() -> None:
    pdf = ToolOutput(
        name="file", url="https://x/r", content_type="application/pdf", content=b"%PDF"
    )
    result = ToolResult(module="convertApi", outputs=(pdf,))
    assert result.image == b"%PDF"
    assert result.extension == ".pdf"
    assert result.data is None


def test_catalog_entry_without_id_is_rejected() -> None:
    with pytest.raises(UnexpectedResponseError):
        _core.tool_from_payload({"name": "no id"})


def test_catalog_parsing_tolerates_missing_optional_fields() -> None:
    tool = _core.tool_from_payload({"id": "bare"})
    assert tool.id == "bare"
    assert tool.name == "bare"
    assert tool.executable is False
    assert tool.handler == "none"
    assert tool.inputs == ()


def test_catalog_input_type_defaults_to_image() -> None:
    tool = _core.tool_from_payload(
        {
            "id": "paletteRemap",
            "inputs": [
                {"id": "in", "required": True},
                {"id": "palette", "type": "palette", "required": True},
            ],
        }
    )
    assert [(i.id, i.type) for i in tool.inputs] == [("in", "image"), ("palette", "palette")]


def test_output_content_type_parameters_are_ignored() -> None:
    response = _core.Response(
        status=200,
        headers=httpx.Headers({"content-type": "image/png; charset=binary"}),
        body=PNG,
    )
    link = _core.OutputLink(name="image", url="https://x/r", content_type="image/png", key="out")
    output = _core.output_from_response(link, response, "pix.AbCdEf12.secret")
    assert output.content_type == "image/png"


def test_encode_params_is_compact_and_strict() -> None:
    assert _core.encode_params(None) == "{}"
    assert _core.encode_params({"a": [1, 2], "b": None}) == '{"a":[1,2],"b":null}'
    with pytest.raises(InvalidInputError):
        _core.encode_params({"a": float("inf")})


def test_catalog_reads_input_and_output_details() -> None:
    tool = _core.tool_from_payload(
        {
            "id": "collage",
            "executable": True,
            "inputs": [
                {
                    "id": "tiles",
                    "description": "Images to lay out.",
                    "multiple": True,
                    "per_file": False,
                    "max_files": 4,
                    "accept": ["image/png", 7],
                }
            ],
            "outputs": [
                {
                    "key": "out",
                    "label": "Image",
                    "description": "The collage.",
                    "kind": "image",
                    "multiple": False,
                    "content_types": ["image/png"],
                },
                {"label": "no key, skipped"},
            ],
        }
    )
    tiles = tool.inputs[0]
    assert (tiles.description, tiles.multiple, tiles.per_file, tiles.max_files) == (
        "Images to lay out.",
        True,
        False,
        4,
    )
    assert tiles.accept == ("image/png",)
    assert len(tool.outputs) == 1
    out = tool.outputs[0]
    assert (out.key, out.kind, out.content_types) == ("out", "image", ("image/png",))
    assert tool.primary_input is tiles


def test_result_output_is_found_by_key_then_name() -> None:
    first = ToolOutput(
        name="image", url="https://x/1", content_type="image/png", content=b"1", key="out#0"
    )
    second = ToolOutput(
        name="image",
        url="https://x/2",
        content_type="image/png",
        content=b"2",
        key="out#1",
        bytes=1,
    )
    result = ToolResult(module="flipBatch", outputs=(first, second))
    assert result.output("out#1") is second
    assert result.output("image") is first
    assert result.output("missing") is None
    assert second.bytes == 1


def _error(status: int, body: object, headers: dict = {}) -> object:  # noqa: B006
    import json

    response = _core.Response(
        status=status, headers=httpx.Headers(headers), body=json.dumps(body).encode()
    )
    return _core.error_from_response(response, "pix.AbCdEf12.secret")


def test_error_details_are_exposed() -> None:
    too_large = _error(413, {"error": "too big", "code": "file_too_large", "max_bytes": 10485760})
    assert isinstance(too_large, FileTooLargeError) and too_large.max_bytes == 10485760

    pixels = _error(422, {"error": "big", "code": "image_too_large", "max_pixels": 40000000})
    assert pixels.max_pixels == 40000000  # type: ignore[attr-defined]

    batch = _error(403, {"error": "batch", "code": "batch_limit", "limit": 10})
    assert isinstance(batch, PermissionDeniedError) and batch.limit == 10

    credits = _error(402, {"error": "no", "code": "insufficient_credits", "required": 12})
    assert isinstance(credits, InsufficientCreditsError) and credits.required_credits == 12

    limited = _error(
        429,
        {
            "error": "slow",
            "code": "rate_limited",
            "retry_after": 7,
            "reset_at": "2026-10-13T10:00:00Z",
        },
    )
    assert isinstance(limited, RateLimitError)
    assert limited.retry_after == 7.0 and limited.reset_at == "2026-10-13T10:00:00Z"

    busy = _error(503, {"error": "busy", "code": "busy"}, {"retry-after": "5"})
    assert isinstance(busy, ServiceUnavailableError) and busy.retry_after == 5.0


def test_plan_transport_chooses_direct_or_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_core, "DIRECT_UPLOAD_MAX_BYTES", 100)
    tool = _core.tool_from_payload(
        {
            "id": "mask",
            "executable": True,
            "inputs": [{"id": "image", "required": True}, {"id": "mask", "required": True}],
        }
    )
    plan = _core.plan_run(
        "mask", tool, image=None, inputs={"image": b"a" * 90, "mask": b"b" * 10}, params=None
    )
    small = [prepare_file(src, form_name=name, max_bytes=LIMIT) for name, src in plan.files]
    assert _core.plan_transport(plan, small).upload is None

    plan = _core.plan_run(
        "mask", tool, image=None, inputs={"image": b"a" * 500, "mask": b"b" * 10}, params=None
    )
    big = [prepare_file(src, form_name=name, max_bytes=LIMIT) for name, src in plan.files]
    transport = _core.plan_transport(plan, big)
    assert transport.upload is big[0] and transport.direct == (big[1],)

    plan = _core.plan_run(
        "mask", tool, image=None, inputs={"image": b"a" * 10, "mask": b"b" * 500}, params=None
    )
    secondary = [prepare_file(src, form_name=name, max_bytes=LIMIT) for name, src in plan.files]
    with pytest.raises(InvalidInputError, match="other inputs"):
        _core.plan_transport(plan, secondary)


def test_read_floor_raises_only_short_read_timeouts() -> None:
    short = httpx.Timeout(5.0)
    floored = _core.with_read_floor(short, 70.0)
    assert (floored.connect, floored.read, floored.write) == (5.0, 70.0, 5.0)
    assert _core.with_read_floor(httpx.Timeout(5.0, read=200.0), 70.0).read == 200.0
    assert _core.with_read_floor(httpx.Timeout(None), 70.0).read is None


def test_timestamps_are_parsed_in_any_rfc3339_form() -> None:
    expected = datetime(2026, 10, 13, 10, 0, tzinfo=timezone.utc).timestamp()
    assert _core.parse_timestamp("2026-10-13T10:00:00Z") == expected
    assert _core.parse_timestamp("2026-10-13T12:00:00+02:00") == expected
    assert _core.parse_timestamp("2026-10-13T10:00:00.123456789Z") == pytest.approx(
        expected + 0.123456
    )
    assert _core.parse_timestamp("2026-10-13T10:00:00.5Z") == pytest.approx(expected + 0.5)
    assert _core.parse_timestamp("2026-10-13T10:00:00.12345Z") == pytest.approx(expected + 0.12345)
    assert _core.parse_timestamp("2026-10-13T12:00:00.25+02:00") == pytest.approx(expected + 0.25)
    assert _core.parse_timestamp("tomorrow") is None
    assert _core.parse_timestamp(None) is None


def _receipt_with(count: int) -> _core.Response:
    entry = '{"key":"r#0","name":"image","url":"https://files.pixlub.com/r/x.png?sig=s"}'
    body = '{"id":"run","outputs":[' + ",".join([entry] * count) + "]}"
    return _core.Response(status=200, headers=httpx.Headers(), body=body.encode())


def test_receipt_reads_a_full_batch_of_four_output_runs() -> None:
    assert _core.MAX_OUTPUTS == 256
    receipt = _core.receipt_from_response(
        _receipt_with(256), "pix.AbCdEf12.secret", "https://api.pixlub.com"
    )
    assert len(receipt.outputs) == 256
    with pytest.raises(UnexpectedResponseError, match="257 outputs"):
        _core.receipt_from_response(
            _receipt_with(257), "pix.AbCdEf12.secret", "https://api.pixlub.com"
        )


def test_retry_hint_uses_the_header_then_the_body() -> None:
    def answer(headers: dict, body: bytes) -> _core.Response:
        return _core.Response(status=429, headers=httpx.Headers(headers), body=body)

    assert _core.retry_hint(answer({"retry-after": "3"}, b'{"retry_after": 9}')) == 3.0
    assert _core.retry_hint(answer({}, b'{"retry_after": 9}')) == 9.0
    assert _core.retry_hint(answer({}, b"not json")) is None


def test_upload_ids_use_the_same_alphabet_as_the_go_sdk() -> None:
    def ticket(upload_id: str) -> _core.UploadTicket:
        body = (
            '{"upload_id":"' + upload_id + '","upload_url":"https://files.pixlub.com/u/x?sig=s",'
            '"method":"PUT"}'
        )
        resp = _core.Response(status=201, headers=httpx.Headers(), body=body.encode())
        return _core.upload_ticket(resp, "pix.AbCdEf12.secret", "https://api.pixlub.com")

    assert ticket("up.1:a-b_c").upload_id == "up.1:a-b_c"
    with pytest.raises(UnexpectedResponseError):
        ticket("up/1")

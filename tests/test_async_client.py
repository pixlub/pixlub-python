from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List

import httpx
import pytest
from conftest import (
    API_KEY,
    CATALOG,
    FILES_HOST,
    IMAGE_PATH,
    PNG,
    RUN_ID,
    json_response,
    parse_multipart,
    run_response,
)

import pixlub
from pixlub.errors import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    ConfigurationError,
    FileTooLargeError,
    InsufficientCreditsError,
    InvalidInputError,
    PermissionDeniedError,
    ServerError,
    ServiceUnavailableError,
    UnknownToolError,
)

_real_sleep = asyncio.sleep


def run(coro):
    return asyncio.run(coro)


def test_async_list_and_schema(make_async_client, api) -> None:
    async def scenario() -> None:
        async with make_async_client() as client:
            tools = await client.tools.list(executable_only=True)
            schema = await client.tools.schema("removeBgApi")
            assert "trimVideo" not in [t.id for t in tools]
            assert schema["type"] == "object"

    run(scenario())
    assert api.requests[0].headers["X-API-Key"] == API_KEY


def test_async_run_returns_the_image(make_async_client, api, tmp_path: Path) -> None:
    source = tmp_path / "photo.png"
    source.write_bytes(PNG)

    async def scenario():
        async with make_async_client(max_retries=0) as client:
            return await client.tools.run("removeBgApi", image=source, params={"a": 1})

    result = run(scenario())

    assert result.image == PNG
    assert result.credits_balance == 41
    fields = parse_multipart(api.calls("POST", "/v1/modules/removeBgApi")[0])
    assert fields["params"]["content"] == b'{"a":1}'
    assert fields["image"]["filename"] == "photo.png"


def test_async_multi_input_run(make_async_client, api) -> None:
    async def scenario() -> None:
        async with make_async_client() as client:
            await client.tools.run("mask", inputs={"image": PNG, "mask": PNG})

    run(scenario())
    fields = parse_multipart(api.calls("POST", "/v1/modules/mask")[0])
    assert set(fields) == {"params", "image", "mask"}


def test_async_local_refusals_send_nothing(make_async_client, api) -> None:
    async def scenario() -> None:
        async with make_async_client() as client:
            with pytest.raises(InvalidInputError):
                await client.tools.run("mask", image=PNG)
            with pytest.raises(UnknownToolError):
                await client.tools.run("notInCatalog", image=PNG)
            with pytest.raises(UnknownToolError):
                await client.tools.run("notInCatalog", inputs={"in": PNG})
            with pytest.raises(InvalidInputError):
                await client.tools.run("../admin/x", image=PNG)

    run(scenario())
    assert api.calls("POST", "/v1/modules/mask") == []
    assert api.calls("POST", "/v1/modules/notInCatalog") == []


def test_async_cache_miss_fetches_the_catalog_once_more(make_async_client, api) -> None:
    newer = {**CATALOG[0], "id": "convertApi", "inputs": [{"id": "in", "required": True}]}

    async def scenario() -> None:
        async with make_async_client() as client:
            await client.tools.run("removeBgApi", image=PNG)
            with pytest.raises(UnknownToolError):
                await client.tools.run("notInCatalog", inputs={"in": PNG})
            api.queue("GET", "/v1/modules", json_response([*CATALOG, newer]))
            await client.tools.run("convertApi", image=PNG)

    run(scenario())
    assert len(api.calls("GET", "/v1/modules")) == 3
    assert api.calls("POST", "/v1/modules/notInCatalog") == []
    assert set(parse_multipart(api.calls("POST", "/v1/modules/convertApi")[0])) == {"params", "in"}


def test_async_errors_are_typed(make_async_client, api) -> None:
    api.queue(
        "POST",
        "/v1/modules/removeBgApi",
        json_response(
            {"error": "not enough", "code": "insufficient_credits", "required": 3}, status=402
        ),
    )

    async def scenario() -> None:
        async with make_async_client(max_retries=0) as client:
            await client.tools.run("removeBgApi", image=PNG)

    with pytest.raises(InsufficientCreditsError) as info:
        run(scenario())
    assert info.value.required_credits == 3


def test_async_get_is_retried_and_post_is_not(make_async_client, api, waits: List[float]) -> None:
    api.queue(
        "GET", "/v1/modules", json_response({"error": "busy"}, status=503), json_response(CATALOG)
    )
    api.queue("POST", "/v1/modules/removeBgApi", json_response({"error": "boom"}, status=500))

    async def scenario() -> None:
        async with make_async_client() as client:
            assert len(await client.tools.list()) == len(CATALOG)
            with pytest.raises(ServerError):
                await client.tools.run("removeBgApi", image=PNG)

    run(scenario())
    assert len(api.calls("GET", "/v1/modules")) == 2
    assert len(api.calls("POST", "/v1/modules/removeBgApi")) == 1
    assert len(waits) == 1


def test_async_timeout_after_send_is_not_retried(make_async_client, api) -> None:
    api.queue("POST", "/v1/modules/removeBgApi", httpx.ReadTimeout("slow"))

    async def scenario() -> None:
        async with make_async_client() as client:
            await client.tools.run("removeBgApi", image=PNG)

    with pytest.raises(APITimeoutError) as info:
        run(scenario())
    assert info.value.request_sent is True
    assert len(api.calls("POST", "/v1/modules/removeBgApi")) == 1


def test_async_connection_refused_is_retried_for_post(make_async_client, api) -> None:
    api.queue(
        "POST",
        "/v1/modules/removeBgApi",
        httpx.ConnectError("refused"),
    )

    async def scenario():
        async with make_async_client() as client:
            return await client.tools.run("removeBgApi", image=PNG)

    assert run(scenario()).image == PNG
    assert len(api.calls("POST", "/v1/modules/removeBgApi")) == 2


def test_async_connection_error_reports_request_not_sent(make_async_client, api) -> None:
    down = httpx.ConnectError("refused")
    api.queue("GET", "/v1/modules", down, down, down)

    async def scenario() -> None:
        async with make_async_client(max_retries=2) as client:
            await client.tools.list()

    with pytest.raises(APIConnectionError) as info:
        run(scenario())
    assert info.value.request_sent is False
    assert info.value.__cause__ is None and info.value.__context__ is None


def test_async_auth_error_message_hides_key(make_async_client, api) -> None:
    api.queue("GET", "/v1/modules", json_response({"error": f"unknown key {API_KEY}"}, status=401))

    async def scenario() -> None:
        async with make_async_client(max_retries=0) as client:
            await client.tools.list()

    with pytest.raises(AuthenticationError) as info:
        run(scenario())
    assert API_KEY not in str(info.value)


def test_async_client_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIXLUB_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        pixlub.AsyncClient()


def test_async_client_repr_hides_secret(make_async_client) -> None:
    client = make_async_client()
    assert "Secret0123456789" not in repr(client)
    assert repr(client).startswith("AsyncClient(")


def test_async_too_large_input_refused_before_upload(make_async_client, api) -> None:
    async def scenario() -> None:
        async with make_async_client(max_input_bytes=10) as client:
            await client.tools.run("removeBgApi", image=PNG)

    with pytest.raises(InvalidInputError):
        run(scenario())
    assert api.calls("POST", "/v1/modules/removeBgApi") == []


def test_async_passed_in_client_is_not_closed(api) -> None:
    http = httpx.AsyncClient(transport=httpx.MockTransport(api))

    async def scenario() -> None:
        async with pixlub.AsyncClient(
            API_KEY, base_url="https://api.pixlub.test", http_client=http
        ):
            pass

    run(scenario())
    assert not http.is_closed
    run(http.aclose())


def test_async_catalog_reads_the_new_fields(make_async_client) -> None:
    async def scenario():
        async with make_async_client() as client:
            return await client.tools.retrieve("collage")

    tool = run(scenario())
    tiles = tool.inputs[1]
    assert (tiles.id, tiles.multiple, tiles.max_files, tiles.per_file) == ("tiles", True, 4, False)
    assert tiles.accept[0] == "image/png" and tiles.description == "The tiles."
    assert tool.outputs[0].key == "out" and tool.outputs[0].kind == "image"
    assert tool.primary_input is not None and tool.primary_input.id == "background"


def test_async_error_details(make_async_client, api) -> None:
    api.queue(
        "POST",
        "/v1/modules/removeBgApi",
        json_response({"error": "too big", "code": "file_too_large", "max_bytes": 52428800}, 413),
        json_response({"error": "busy", "code": "busy"}, 503, {"retry-after": "5"}),
    )

    async def scenario():
        async with make_async_client(max_retries=0) as client:
            with pytest.raises(FileTooLargeError) as too_large:
                await client.tools.run("removeBgApi", image=PNG)
            with pytest.raises(ServiceUnavailableError) as busy:
                await client.tools.run("removeBgApi", image=PNG)
            return too_large.value, busy.value

    too_large, busy = run(scenario())
    assert too_large.max_bytes == 52428800
    assert busy.retry_after == 5.0 and busy.code == "busy"


def test_async_downloads_run_a_few_at_a_time(api) -> None:
    outputs = [
        {"key": f"out#{i}", "name": "image", "url": f"https://{FILES_HOST}/r/{i}_result.png"}
        for i in range(10)
    ]
    api.queue("POST", "/v1/modules/flipBatch", run_response(outputs))
    state = {"running": 0, "peak": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == FILES_HOST:
            state["running"] += 1
            state["peak"] = max(state["peak"], state["running"])
            await _real_sleep(0.005)
            state["running"] -= 1
        return api(request)

    async def scenario():
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with pixlub.AsyncClient(
            API_KEY, base_url="https://api.pixlub.test", http_client=http
        ) as c:
            result = await c.tools.run("flipBatch", image=[PNG] * 10)
        await http.aclose()
        return result

    result = run(scenario())
    assert [o.key for o in result.outputs] == [f"out#{i}" for i in range(10)]
    assert result.output("out#3") is result.outputs[3]
    assert 1 < state["peak"] <= 4


def test_async_failed_download_says_the_run_was_charged(make_async_client, api) -> None:
    api.queue("GET", IMAGE_PATH, json_response({"error": "invalid or expired link"}, status=403))

    async def scenario() -> None:
        async with make_async_client(max_retries=0) as client:
            await client.tools.run("removeBgApi", image=PNG)

    with pytest.raises(PermissionDeniedError) as info:
        run(scenario())
    assert RUN_ID in str(info.value) and "charged" in str(info.value)


def test_async_run_streams_a_file_from_disk(make_async_client, api, tmp_path: Path) -> None:
    source = tmp_path / "in.png"
    source.write_bytes(PNG * 5000)

    async def scenario() -> None:
        async with make_async_client() as client:
            await client.tools.run("removeBgApi", image=source)

    run(scenario())
    request = api.calls("POST", "/v1/modules/removeBgApi")[0]
    assert parse_multipart(request)["image"]["content"] == PNG * 5000
    assert "transfer-encoding" not in request.headers

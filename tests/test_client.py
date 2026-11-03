from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest
from conftest import (
    API_KEY,
    BASE_URL,
    CATALOG,
    DATA_LINK,
    IMAGE_LINK,
    IMAGE_PATH,
    PNG,
    RUN_ID,
    json_response,
    parse_multipart,
    run_response,
)

import pixlub
from pixlub import _core
from pixlub.errors import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    ConfigurationError,
    InsufficientCreditsError,
    InvalidInputError,
    InvalidRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServerError,
    UnexpectedResponseError,
    UnknownToolError,
)


def test_client_reads_key_and_base_url_from_environment(
    monkeypatch: pytest.MonkeyPatch, api
) -> None:
    monkeypatch.setenv("PIXLUB_API_KEY", API_KEY)
    monkeypatch.setenv("PIXLUB_BASE_URL", "https://env.pixlub.test/")
    client = pixlub.Client(http_client=httpx.Client(transport=httpx.MockTransport(api)))

    client.tools.list()

    request = api.requests[0]
    assert request.url.host == "env.pixlub.test"
    assert request.headers["X-API-Key"] == API_KEY


def test_missing_key_fails_before_any_request(monkeypatch: pytest.MonkeyPatch, api) -> None:
    monkeypatch.delenv("PIXLUB_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="PIXLUB_API_KEY"):
        pixlub.Client()
    assert api.requests == []


@pytest.mark.parametrize(
    "bad_key",
    ["pix.onlyonepart", "sk_live_123", "pix.a.b c", "pix..secret", "  "],
)
def test_malformed_key_is_rejected_without_echoing_it(bad_key: str) -> None:
    with pytest.raises(ConfigurationError) as info:
        pixlub.Client(bad_key)
    assert bad_key.strip() not in str(info.value) or not bad_key.strip()


def test_key_surrounding_whitespace_is_stripped(api) -> None:
    client = pixlub.Client(
        f"  {API_KEY}\n",
        base_url=BASE_URL,
        http_client=httpx.Client(transport=httpx.MockTransport(api)),
    )
    client.tools.list()
    assert api.requests[0].headers["X-API-Key"] == API_KEY


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.example.com", "https://api.example.com"),
        ("https://api.example.com/", "https://api.example.com"),
        ("https://example.com/pixlub/", "https://example.com/pixlub"),
        ("http://localhost:8080", "http://localhost:8080"),
        ("http://127.0.0.1:8080/", "http://127.0.0.1:8080"),
    ],
)
def test_base_url_accepted_forms(url: str, expected: str) -> None:
    assert pixlub.Client(API_KEY, base_url=url).__repr__().count(expected) == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://api.example.com",  # plain HTTP would expose the key
        "ftp://api.example.com",
        "https://user:secret@api.example.com",  # credentials in the URL
        "https://api.example.com/?token=1",
        "https://api.example.com/#frag",
        "https://",
        "https://api.example.com:99999",
        "not a url",
    ],
)
def test_unsafe_base_url_is_rejected(url: str) -> None:
    with pytest.raises(ConfigurationError):
        pixlub.Client(API_KEY, base_url=url)


def test_repr_never_shows_the_secret() -> None:
    client = pixlub.Client(API_KEY, base_url=BASE_URL)
    text = repr(client)
    assert "Secret0123456789" not in text
    assert "pix.AbCdEf12" in text


def test_user_agent_identifies_the_sdk(client, api) -> None:
    client.tools.list()
    assert api.requests[0].headers["User-Agent"].startswith("pixlub-python/")


@pytest.mark.parametrize("value", [0, -1, "2", True])
def test_invalid_timeout_is_rejected(make_client, value: Any) -> None:
    with pytest.raises(ConfigurationError):
        make_client(timeout=value)


@pytest.mark.parametrize("value", [-1, 6, True, 1.5])
def test_invalid_max_retries_is_rejected(make_client, value: Any) -> None:
    with pytest.raises(ConfigurationError):
        make_client(max_retries=value)


def test_passed_in_http_client_is_not_closed(api) -> None:
    http = httpx.Client(transport=httpx.MockTransport(api))
    with pixlub.Client(API_KEY, base_url=BASE_URL, http_client=http):
        pass
    assert not http.is_closed


@pytest.mark.parametrize(
    "tool_id",
    ["../admin/x", "removeBgApi?x=1", "a/b", "", "  ", "x" * 70, None, "pix.x", "a#b"],
)
def test_tool_ids_that_are_not_plain_names_never_reach_the_network(
    client, api, tool_id: Any
) -> None:
    with pytest.raises(InvalidInputError):
        client.tools.schema(tool_id)
    with pytest.raises(InvalidInputError):
        client.tools.run(tool_id, image=PNG)
    assert api.requests == []


def test_list_returns_every_tool(client) -> None:
    tools = client.tools.list()
    assert [t.id for t in tools] == [t["id"] for t in CATALOG]
    assert tools[0].inputs[0].id == "image"
    assert tools[0].inputs[0].required is True


def test_list_can_skip_non_executable_tools(client) -> None:
    ids = [t.id for t in client.tools.list(executable_only=True)]
    assert "trimVideo" not in ids
    assert "removeBgApi" in ids


def test_catalog_is_cached_between_runs(client, api) -> None:
    client.tools.run("removeBgApi", image=PNG)
    client.tools.run("removeBgApi", image=PNG)
    assert len(api.calls("GET", "/v1/modules")) == 1


def test_retrieve_unknown_tool_raises(client) -> None:
    with pytest.raises(UnknownToolError):
        client.tools.retrieve("doesNotExist")


def test_schema_is_returned_as_a_dict(client, api) -> None:
    schema = client.tools.schema("removeBgApi")
    assert schema["type"] == "object"
    assert api.calls("GET", "/v1/modules/removeBgApi/schema")


def test_catalog_must_be_a_list(make_client, api) -> None:
    api.queue("GET", "/v1/modules", json_response({"not": "a list"}))
    with pytest.raises(UnexpectedResponseError):
        make_client().tools.list()


def test_run_sends_key_in_header_and_image_in_multipart(client, api) -> None:
    result = client.tools.run("removeBgApi", image=PNG, params={"width": 512})

    request = api.calls("POST", "/v1/modules/removeBgApi")[0]
    assert request.headers["X-API-Key"] == API_KEY
    assert API_KEY not in str(request.url)
    fields = parse_multipart(request)
    assert fields["image"]["filename"] == "image.png"
    assert fields["image"]["content_type"] == "image/png"
    assert fields["image"]["content"] == PNG
    assert fields["params"]["content"] == b'{"width":512}'

    assert result.image == PNG
    assert result.content_type == "image/png"
    assert result.credits_balance == 41


def test_run_with_a_path_keeps_the_file_name(client, api, tmp_path: Path) -> None:
    source = tmp_path / "vacances photo.jpg"
    source.write_bytes(PNG)

    client.tools.run("removeBgApi", image=source)

    fields = parse_multipart(api.calls("POST", "/v1/modules/removeBgApi")[0])
    assert fields["image"]["filename"] == "vacances photo.jpg"
    assert fields["image"]["content_type"] == "image/png"


def test_run_with_a_binary_file_object(client, api, tmp_path: Path) -> None:
    source = tmp_path / "in.png"
    source.write_bytes(PNG)
    with source.open("rb") as handle:
        client.tools.run("removeBgApi", image=handle)

    fields = parse_multipart(api.calls("POST", "/v1/modules/removeBgApi")[0])
    assert fields["image"]["content"] == PNG
    assert fields["image"]["filename"] == "in.png"


def test_run_sends_multiple_named_inputs(client, api) -> None:
    client.tools.run("mask", inputs={"image": PNG, "mask": PNG})

    fields = parse_multipart(api.calls("POST", "/v1/modules/mask")[0])
    assert set(fields) == {"params", "image", "mask"}


def test_generator_tool_without_inputs_still_sends_multipart(client, api) -> None:
    client.tools.run("checker", params={"size": 8})

    request = api.calls("POST", "/v1/modules/checker")[0]
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert parse_multipart(request)["params"]["content"] == b'{"size":8}'


def test_several_images_need_inputs_not_image(client, api) -> None:
    with pytest.raises(InvalidInputError, match="several images"):
        client.tools.run("mask", image=PNG)
    assert api.calls("POST", "/v1/modules/mask") == []


def test_missing_required_input_is_refused_locally(client, api) -> None:
    with pytest.raises(InvalidInputError, match="missing required input"):
        client.tools.run("mask", inputs={"image": PNG})
    assert api.calls("POST", "/v1/modules/mask") == []


def test_unknown_input_name_is_refused_locally(client, api) -> None:
    with pytest.raises(InvalidInputError, match="no input named 'photo'"):
        client.tools.run("removeBgApi", inputs={"photo": PNG})
    assert api.calls("POST", "/v1/modules/removeBgApi") == []


def test_image_for_a_tool_without_image_input_is_refused(client, api) -> None:
    with pytest.raises(InvalidInputError, match="takes no image"):
        client.tools.run("checker", image=PNG)
    assert api.calls("POST", "/v1/modules/checker") == []


def test_non_executable_tool_is_refused_locally(client, api) -> None:
    with pytest.raises(InvalidInputError, match="cannot be run through the API"):
        client.tools.run("trimVideo", image=PNG)
    assert api.calls("POST", "/v1/modules/trimVideo") == []


@pytest.mark.parametrize("files", [{"image": PNG}, {"inputs": {"in": PNG}}])
def test_tool_missing_from_the_catalog_is_refused_locally(
    client, api, files: Dict[str, Any]
) -> None:
    with pytest.raises(UnknownToolError, match="not in the tool catalog"):
        client.tools.run("notInCatalog", **files)
    assert len(api.calls("GET", "/v1/modules")) == 1
    assert [r for r in api.requests if r.method in ("POST", "PUT")] == []


@pytest.mark.parametrize("files", [{"image": PNG}, {"inputs": {"in": PNG}}])
def test_cache_miss_fetches_the_catalog_once_more(client, api, files: Dict[str, Any]) -> None:
    client.tools.run("removeBgApi", image=PNG)
    with pytest.raises(UnknownToolError):
        client.tools.run("notInCatalog", **files)
    assert len(api.calls("GET", "/v1/modules")) == 2
    assert api.calls("POST", "/v1/modules/notInCatalog") == []


def test_tool_newer_than_the_cached_catalog_is_found(client, api) -> None:
    client.tools.run("removeBgApi", image=PNG)
    newer = {**CATALOG[0], "id": "convertApi", "inputs": [{"id": "in", "required": True}]}
    api.queue("GET", "/v1/modules", json_response([*CATALOG, newer]))

    client.tools.run("convertApi", image=PNG)

    assert len(api.calls("GET", "/v1/modules")) == 2
    assert set(parse_multipart(api.calls("POST", "/v1/modules/convertApi")[0])) == {
        "params",
        "in",
    }


def test_too_large_file_is_refused_before_upload(make_client, api) -> None:
    client = make_client(max_input_bytes=100)
    with pytest.raises(InvalidInputError, match="limit is 100 bytes"):
        client.tools.run("removeBgApi", image=PNG + b"\x00" * 200)
    assert api.calls("POST", "/v1/modules/removeBgApi") == []


def test_missing_file_is_refused_locally(client, api, tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="Cannot read"):
        client.tools.run("removeBgApi", image=tmp_path / "absent.png")
    assert api.calls("POST", "/v1/modules/removeBgApi") == []


def test_directory_is_refused_as_input(client, tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="not a regular file"):
        client.tools.run("removeBgApi", image=tmp_path)


def test_remote_urls_are_not_fetched(client, api) -> None:
    with pytest.raises(InvalidInputError, match="Cannot read"):
        client.tools.run("removeBgApi", image="https://evil.test/image.png")
    assert api.calls("POST", "/v1/modules/removeBgApi") == []
    assert {r.url.host for r in api.requests} == {"api.pixlub.test"}


def test_unsupported_image_type_is_a_type_error(client) -> None:
    with pytest.raises(TypeError):
        client.tools.run("removeBgApi", image=42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "params",
    [{"x": object()}, {"x": float("nan")}, ["not", "a", "dict"]],
)
def test_params_must_be_json_values(client, api, params: Any) -> None:
    with pytest.raises(InvalidInputError):
        client.tools.run("removeBgApi", image=PNG, params=params)
    assert api.calls("POST", "/v1/modules/removeBgApi") == []


def test_result_reads_message_and_credits(client, api) -> None:
    api.queue(
        "POST",
        "/v1/modules/removeBgApi",
        run_response(
            message="Background removed — check the subject", credits={"cost": 10, "balance": 40}
        ),
    )
    result = client.tools.run("removeBgApi", image=PNG)
    assert result.message == "Background removed — check the subject"
    assert result.cost == 10
    assert result.credits_balance == 40
    assert result.id == RUN_ID
    assert result.expires_at == "2026-10-13T10:00:00Z"


def test_run_response_without_outputs_is_rejected(client, api) -> None:
    api.queue("POST", "/v1/modules/removeBgApi", json_response({"id": RUN_ID}))
    with pytest.raises(UnexpectedResponseError, match="no output"):
        client.tools.run("removeBgApi", image=PNG)


def test_outputs_are_downloaded_without_the_key(client, api) -> None:
    client.tools.run("removeBgApi", image=PNG)

    download = api.calls("GET", IMAGE_PATH)[0]
    assert "X-API-Key" not in download.headers
    assert download.url.params["sig"] == "signature"


def test_data_output_is_parsed(client, api) -> None:
    api.queue(
        "POST",
        "/v1/modules/removeBgApi",
        run_response(
            [
                {"name": "image", "url": IMAGE_LINK, "content_type": "image/png"},
                {"name": "data", "url": DATA_LINK, "content_type": "application/json"},
            ]
        ),
    )
    result = client.tools.run("removeBgApi", image=PNG)

    assert result.image == PNG
    assert result.data == {"faces": []}
    assert [o.name for o in result.outputs] == ["image", "data"]
    assert result.output("data").extension == ".json"


def test_data_only_result_has_no_image(client, api, tmp_path: Path) -> None:
    api.queue(
        "POST",
        "/v1/modules/removeBgApi",
        run_response([{"name": "data", "url": DATA_LINK, "content_type": "application/json"}]),
    )
    result = client.tools.run("removeBgApi", image=PNG)

    assert result.image is None
    assert result.data == {"faces": []}
    with pytest.raises(pixlub.InvalidInputError, match="only data"):
        result.save(tmp_path / "out.png")
    with pytest.raises(ValueError, match="only data"):
        result.save(tmp_path / "out.png")


def test_relative_links_are_resolved_against_the_base_url(client, api) -> None:
    relative = f"v1/results/{RUN_ID}_result.png?exp=1&sig=s"
    api.queue("POST", "/v1/modules/removeBgApi", run_response([{"name": "image", "url": relative}]))
    result = client.tools.run("removeBgApi", image=PNG)

    assert result.outputs[0].url == f"{BASE_URL}/{relative}"
    assert result.image == PNG


@pytest.mark.parametrize(
    "link",
    [
        f"http://elsewhere.test/v1/results/{RUN_ID}_result.png",
        f"https://user:pass@api.pixlub.test/v1/results/{RUN_ID}_result.png",
        "ftp://api.pixlub.test/result.png",
    ],
)
def test_unsafe_links_are_never_downloaded(client, api, link: str) -> None:
    api.queue("POST", "/v1/modules/removeBgApi", run_response([{"name": "image", "url": link}]))
    with pytest.raises(UnexpectedResponseError, match="Refused a result link"):
        client.tools.run("removeBgApi", image=PNG)
    assert not [r for r in api.requests if r.method == "GET" and "result" in r.url.path]


def test_expired_link_is_a_permission_error(make_client, api) -> None:
    client = make_client(max_retries=0)
    api.queue(
        "GET",
        IMAGE_PATH,
        json_response({"error": "invalid or expired link"}, status=403),
    )
    with pytest.raises(PermissionDeniedError, match="expired"):
        client.tools.run("removeBgApi", image=PNG)


def test_failed_download_is_retried_without_running_again(make_client, api) -> None:
    client = make_client(max_retries=2)
    api.queue(
        "GET",
        IMAGE_PATH,
        json_response({"error": "busy"}, status=503),
        httpx.ReadTimeout("slow"),
    )
    result = client.tools.run("removeBgApi", image=PNG)

    assert result.image == PNG
    assert len(api.calls("POST", "/v1/modules/removeBgApi")) == 1
    assert len(api.calls("GET", IMAGE_PATH)) == 3


def test_oversized_output_is_not_read(make_client, api, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_core, "MAX_OUTPUT_BYTES", 16)
    client = make_client()
    with pytest.raises(UnexpectedResponseError, match="larger than"):
        client.tools.run("removeBgApi", image=PNG)


def test_oversized_catalog_is_not_read(make_client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_core, "MAX_JSON_BYTES", 10)
    with pytest.raises(UnexpectedResponseError):
        make_client().tools.list()


def test_timeout_override_is_passed_to_the_request(client, api) -> None:
    client.tools.run("removeBgApi", image=PNG, timeout=7.5)
    run = api.calls("POST", "/v1/modules/removeBgApi")[0].extensions["timeout"]
    download = api.calls("GET", IMAGE_PATH)[0].extensions["timeout"]
    assert run["connect"] == 7.5
    assert download["read"] == 7.5
    assert run["read"] == 70.0


def test_a_longer_read_timeout_is_kept(client, api) -> None:
    client.tools.run("removeBgApi", image=PNG, timeout=httpx.Timeout(5.0, read=300.0))
    assert api.calls("POST", "/v1/modules/removeBgApi")[0].extensions["timeout"]["read"] == 300.0


def test_default_base_url_is_the_public_api(monkeypatch: pytest.MonkeyPatch, api) -> None:
    monkeypatch.delenv("PIXLUB_BASE_URL", raising=False)
    client = pixlub.Client(API_KEY, http_client=httpx.Client(transport=httpx.MockTransport(api)))
    client.tools.list()
    assert str(api.requests[0].url) == "https://api.pixlub.com/v1/modules"
    assert pixlub.DEFAULT_BASE_URL == "https://api.pixlub.com"


def test_save_refuses_to_overwrite_by_default(client, tmp_path: Path) -> None:
    result = client.tools.run("removeBgApi", image=PNG)
    target = tmp_path / f"out{result.extension}"

    assert result.save(target) == target
    assert target.read_bytes() == PNG
    with pytest.raises(FileExistsError):
        result.save(target)
    result.save(target, overwrite=True)


@pytest.mark.parametrize(
    ("status", "body", "headers", "expected", "code"),
    [
        (
            400,
            {"error": "Invalid parameters", "details": [{"keyword": "type"}]},
            {},
            InvalidRequestError,
            None,
        ),
        (401, {"error": "API key required: X-API-Key header"}, {}, AuthenticationError, None),
        (
            402,
            {
                "error": "not enough credits for this module",
                "code": "insufficient_credits",
                "required": 5,
            },
            {},
            InsufficientCreditsError,
            "insufficient_credits",
        ),
        (
            403,
            {"error": "api keys require a paid plan", "code": "plan_required"},
            {},
            PermissionDeniedError,
            "plan_required",
        ),
        (404, {"error": "Unknown module: nope"}, {}, NotFoundError, None),
        (429, {"error": "Too many requests"}, {"retry-after": "3"}, RateLimitError, None),
        (500, {"error": "Process processing failed"}, {}, ServerError, None),
        (503, {"error": "Billing unavailable"}, {}, ServerError, None),
        (418, {"error": "teapot"}, {}, InvalidRequestError, None),
    ],
)
def test_error_responses_map_to_typed_exceptions(
    make_client,
    api,
    status: int,
    body: Dict[str, Any],
    headers: Dict[str, str],
    expected: type,
    code,
) -> None:
    client = make_client(max_retries=0)
    api.queue(
        "POST", "/v1/modules/removeBgApi", json_response(body, status=status, headers=headers)
    )

    with pytest.raises(expected) as info:
        client.tools.run("removeBgApi", image=PNG)

    error = info.value
    assert isinstance(error, APIError)
    assert error.http_status == status
    assert error.code == code
    assert str(status) in str(error)


def test_insufficient_credits_exposes_the_required_amount(make_client, api) -> None:
    client = make_client(max_retries=0)
    api.queue(
        "POST",
        "/v1/modules/removeBgApi",
        json_response(
            {"error": "not enough", "code": "insufficient_credits", "required": 5}, status=402
        ),
    )
    with pytest.raises(InsufficientCreditsError) as info:
        client.tools.run("removeBgApi", image=PNG)
    assert info.value.required_credits == 5


def test_invalid_params_expose_details(make_client, api) -> None:
    client = make_client(max_retries=0)
    details = [{"instancePath": "/width", "message": "must be number"}]
    api.queue(
        "POST",
        "/v1/modules/removeBgApi",
        json_response({"error": "Invalid parameters", "details": details}, status=400),
    )
    with pytest.raises(InvalidRequestError) as info:
        client.tools.run("removeBgApi", image=PNG, params={"width": "wide"})
    assert info.value.details == details


def test_framework_not_found_uses_its_precise_message(make_client, api) -> None:
    client = make_client(max_retries=0)
    api.queue(
        "GET",
        "/v1/modules",
        json_response(
            {"message": "Route GET:/v1/modules not found", "error": "Not Found", "statusCode": 404},
            status=404,
        ),
    )
    with pytest.raises(NotFoundError, match="Route GET:/v1/modules not found"):
        client.tools.list()


def test_error_text_never_contains_the_key(make_client, api) -> None:
    client = make_client(max_retries=0)
    api.queue(
        "POST",
        "/v1/modules/removeBgApi",
        json_response({"error": f"bad key {API_KEY}"}, status=401),
    )
    with pytest.raises(AuthenticationError) as info:
        client.tools.run("removeBgApi", image=PNG)
    assert API_KEY not in str(info.value)
    assert "Secret0123456789" not in str(info.value)


def test_redirects_are_never_followed(make_client, api) -> None:
    client = make_client(max_retries=0)
    api.queue(
        "GET",
        "/v1/modules",
        httpx.Response(302, headers={"location": "https://attacker.test/steal"}),
    )
    with pytest.raises(APIError) as info:
        client.tools.list()
    assert info.value.http_status == 302
    assert [r.url.host for r in api.requests] == ["api.pixlub.test"]


def test_get_is_retried_on_503(make_client, api, waits: List[float]) -> None:
    client = make_client()
    api.queue(
        "GET", "/v1/modules", json_response({"error": "busy"}, status=503), json_response(CATALOG)
    )

    assert len(client.tools.list()) == len(CATALOG)
    assert len(api.calls("GET", "/v1/modules")) == 2
    assert len(waits) == 1


def test_post_is_not_retried_after_a_server_error(make_client, api) -> None:
    client = make_client()
    api.queue("POST", "/v1/modules/removeBgApi", json_response({"error": "boom"}, status=500))

    with pytest.raises(ServerError):
        client.tools.run("removeBgApi", image=PNG)
    assert len(api.calls("POST", "/v1/modules/removeBgApi")) == 1


def test_post_is_retried_on_429_and_honours_retry_after(
    make_client, api, waits: List[float]
) -> None:
    client = make_client()
    api.queue(
        "POST",
        "/v1/modules/removeBgApi",
        json_response({"error": "slow down"}, 429, {"retry-after": "2"}),
    )

    result = client.tools.run("removeBgApi", image=PNG)

    assert result.image == PNG
    assert len(api.calls("POST", "/v1/modules/removeBgApi")) == 2
    assert waits == [2.0]


def test_retry_after_is_capped(make_client, api, waits: List[float]) -> None:
    client = make_client()
    api.queue(
        "GET",
        "/v1/modules",
        json_response({"error": "x"}, 429, {"retry-after": "100000"}),
        json_response(CATALOG),
    )
    client.tools.list()
    assert waits == [60.0]


def test_post_is_retried_when_the_connection_was_refused(make_client, api) -> None:
    client = make_client()
    refused = httpx.ConnectError("connection refused")
    api.queue("POST", "/v1/modules/removeBgApi", refused, refused)

    result = client.tools.run("removeBgApi", image=PNG)

    assert result.image == PNG
    assert len(api.calls("POST", "/v1/modules/removeBgApi")) == 3


def test_post_read_timeout_is_not_retried(make_client, api) -> None:
    client = make_client()
    api.queue("POST", "/v1/modules/removeBgApi", httpx.ReadTimeout("slow"))

    with pytest.raises(APITimeoutError) as info:
        client.tools.run("removeBgApi", image=PNG)

    assert info.value.request_sent is True
    assert "check your credit balance" in str(info.value)
    assert len(api.calls("POST", "/v1/modules/removeBgApi")) == 1


def test_get_read_timeout_is_retried(make_client, api) -> None:
    client = make_client()
    api.queue("GET", "/v1/modules", httpx.ReadTimeout("slow"), json_response(CATALOG))
    assert len(client.tools.list()) == len(CATALOG)


def test_connection_failure_after_retries_reports_it_was_not_sent(make_client, api) -> None:
    client = make_client(max_retries=1)
    down = httpx.ConnectError("refused")
    api.queue("GET", "/v1/modules", down, down)

    with pytest.raises(APIConnectionError) as info:
        client.tools.list()

    assert not isinstance(info.value, APITimeoutError)
    assert info.value.request_sent is False
    assert len(api.calls("GET", "/v1/modules")) == 2


def test_no_retries_when_max_retries_is_zero(make_client, api) -> None:
    client = make_client(max_retries=0)
    api.queue("GET", "/v1/modules", httpx.ConnectError("refused"))

    with pytest.raises(APIConnectionError):
        client.tools.list()
    assert len(api.calls("GET", "/v1/modules")) == 1


def test_transport_errors_do_not_chain_the_request(make_client, api) -> None:
    client = make_client(max_retries=0)
    api.queue("GET", "/v1/modules", httpx.ConnectError("refused"))

    with pytest.raises(APIConnectionError) as info:
        client.tools.list()

    assert info.value.__cause__ is None
    assert info.value.__context__ is None


def test_retry_after_is_read_from_the_body_without_a_header(
    make_client, api, waits: List[float]
) -> None:
    client = make_client()
    api.queue(
        "POST",
        "/v1/modules/removeBgApi",
        json_response({"error": "slow down", "code": "rate_limited", "retry_after": 4}, 429),
    )

    client.tools.run("removeBgApi", image=PNG)

    assert waits == [4.0]


def test_unreadable_run_answer_says_it_was_charged(client, api) -> None:
    api.queue("POST", "/v1/modules/removeBgApi", json_response({"id": RUN_ID}))

    with pytest.raises(UnexpectedResponseError) as caught:
        client.tools.run("removeBgApi", image=PNG)

    assert RUN_ID in str(caught.value)
    assert "charged" in str(caught.value)

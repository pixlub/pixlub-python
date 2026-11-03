from __future__ import annotations

import os

import pytest

import pixlub

BASE_URL = os.environ.get("PIXLUB_BASE_URL")
API_KEY = os.environ.get("PIXLUB_API_KEY")
PLACEHOLDER_KEY = "pix.AbCdEf12.LiveTestOnlyNotARealSecret0123456789"

pytestmark = pytest.mark.skipif(not BASE_URL, reason="PIXLUB_BASE_URL is not set")


def _client() -> pixlub.Client:
    return pixlub.Client(API_KEY or PLACEHOLDER_KEY, base_url=BASE_URL)


def test_catalog_lists_tools() -> None:
    with _client() as client:
        tools = client.tools.list()
    assert tools, "the server returned an empty catalog"
    assert any(tool.id == "removeBgApi" for tool in tools)


def test_schema_of_a_known_tool() -> None:
    with _client() as client:
        schema = client.tools.schema("flip")
    assert schema.get("type") == "object"
    assert "axis" in schema["properties"]


def test_unknown_tool_schema_is_not_found() -> None:
    with _client() as client, pytest.raises(pixlub.NotFoundError):
        client.tools.schema("doesNotExistAnywhere")


@pytest.mark.skipif(not API_KEY, reason="PIXLUB_API_KEY is not set")
def test_run_a_local_tool_on_a_real_image() -> None:
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )

    with _client() as client:
        result = client.tools.run("flip", image=png, params={"axis": "h"})

    assert result.content_type == "image/png"
    assert result.image.startswith(b"\x89PNG")
    assert result.credits_balance is not None

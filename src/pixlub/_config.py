"""API key and base URL."""

from __future__ import annotations

import os
import re
from typing import Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

from .errors import ConfigurationError

API_KEY_ENV_VAR = "PIXLUB_API_KEY"
BASE_URL_ENV_VAR = "PIXLUB_BASE_URL"
DEFAULT_BASE_URL = "https://api.pixlub.com"

_API_KEY_RE = re.compile(r"pix\.[A-Za-z0-9_-]{1,64}\.[A-Za-z0-9_-]{1,256}")
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def resolve_api_key(api_key: Optional[str]) -> str:
    raw = api_key if api_key is not None else os.environ.get(API_KEY_ENV_VAR)
    key = raw.strip() if raw else ""
    if not key:
        raise ConfigurationError(
            "No API key provided. Pass api_key=... or set the "
            f"{API_KEY_ENV_VAR} environment variable."
        )
    if not _API_KEY_RE.fullmatch(key):
        raise ConfigurationError(
            "The API key is malformed. Copy the whole key (pix.<prefix>.<secret>) "
            "from your dashboard."
        )
    return key


def resolve_base_url(base_url: Optional[str]) -> str:
    raw = base_url if base_url is not None else os.environ.get(BASE_URL_ENV_VAR)
    url = (raw or DEFAULT_BASE_URL).strip()
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018 - raises ValueError on an invalid port
    except ValueError:
        raise ConfigurationError(
            "Invalid base URL. Expected a form like https://api.example.com"
        ) from None

    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConfigurationError(
            "The base URL must start with http:// or https:// and name a host."
        )
    if parts.username is not None or parts.password is not None:
        raise ConfigurationError("Do not put credentials in the base URL. Use api_key instead.")
    if parts.query or parts.fragment:
        raise ConfigurationError("The base URL must not contain a query string or a fragment.")
    if parts.scheme == "http" and parts.hostname not in _LOOPBACK_HOSTS:
        raise ConfigurationError(
            "Plain http:// is only accepted for localhost. Use https:// so your API key "
            "is encrypted in transit."
        )

    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def resolve_download_url(url: str, base_url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("empty link")
    absolute = urljoin(base_url + "/", url.strip())
    parts = urlsplit(absolute)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("link is not an http(s) URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError("link carries credentials")
    if parts.scheme == "http" and parts.hostname not in _LOOPBACK_HOSTS:
        raise ValueError("link uses plain http outside localhost")
    return absolute


def key_mask(api_key: str) -> str:
    return f"pix.{api_key.split('.')[1]}.***"


def redact(text: str, api_key: str) -> str:
    return text.replace(api_key, key_mask(api_key))

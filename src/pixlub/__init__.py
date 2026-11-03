"""Official Python SDK for the Pixlub API::

import pixlub

client = pixlub.Client()  # reads PIXLUB_API_KEY
result = client.tools.run("removeBgApi", image="photo.jpg")
result.save(f"photo-nobg{result.extension}")
"""

from ._config import API_KEY_ENV_VAR, BASE_URL_ENV_VAR, DEFAULT_BASE_URL
from ._core import DIRECT_UPLOAD_MAX_BYTES, MAX_INPUT_FILES
from ._files import SourceURL
from ._version import __version__
from .async_client import AsyncClient
from .client import Client
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

__all__ = [
    "API_KEY_ENV_VAR",
    "BASE_URL_ENV_VAR",
    "DEFAULT_BASE_URL",
    "DIRECT_UPLOAD_MAX_BYTES",
    "MAX_INPUT_FILES",
    "APIConnectionError",
    "APIError",
    "APITimeoutError",
    "AsyncClient",
    "AuthenticationError",
    "Client",
    "ConfigurationError",
    "FileTooLargeError",
    "InsufficientCreditsError",
    "InvalidInputError",
    "InvalidRequestError",
    "JobTimeoutError",
    "NotFoundError",
    "PermissionDeniedError",
    "PixlubError",
    "RateLimitError",
    "ServerError",
    "ServiceUnavailableError",
    "SourceURL",
    "Tool",
    "ToolInput",
    "ToolOutput",
    "ToolOutputSpec",
    "ToolResult",
    "UnexpectedResponseError",
    "UnknownToolError",
    "__version__",
]

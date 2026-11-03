"""Objects returned by the SDK."""

from __future__ import annotations

import builtins
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from ._files import extension_for
from .errors import InvalidInputError


@dataclass(frozen=True)
class ToolInput:
    """A file input of a tool. ``id`` is the name to use with ``inputs={id: ...}``."""

    id: str
    label: str
    required: bool
    type: str = "image"
    """``image``, ``channel`` (read as grayscale) or ``batch``."""
    description: str = ""
    multiple: bool = False
    """True when the input takes several files: pass a list, such as ``inputs={id: [a, b]}``."""
    per_file: bool = False
    """True when each file is a separate run, charged separately (a batch)."""
    max_files: Optional[int] = None
    """Most files this input takes in one call, for your plan. None when not given."""
    accept: Tuple[str, ...] = ()
    """Content types the input accepts, such as ``("image/png", "image/jpeg")``."""


@dataclass(frozen=True)
class ToolOutputSpec:
    """An output a tool can produce, as described by the catalog."""

    key: str
    """Matches ``ToolOutput.key`` in a result. In a batch, the result keys get ``#0``, ``#1``..."""
    label: str = ""
    description: str = ""
    kind: str = ""
    """What the output holds, such as ``image``, ``data`` or ``file``."""
    multiple: bool = False
    content_types: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Tool:
    """A tool (module) from the catalog returned by ``client.tools.list()``."""

    id: str
    name: str
    description: str
    category: str
    executable: bool
    """False when the tool cannot be run through the API."""
    handler: str
    """``local``, ``remote`` or ``none``."""
    inputs: Tuple[ToolInput, ...] = ()
    outputs: Tuple[ToolOutputSpec, ...] = ()

    @property
    def primary_input(self) -> Optional[ToolInput]:
        """The main input, filled by a ``SourceURL``: the first required one, else the first."""
        for item in self.inputs:
            if item.required:
                return item
        return self.inputs[0] if self.inputs else None


def _write(path: Union[str, "os.PathLike[str]"], content: bytes, overwrite: bool) -> Path:
    target = Path(path)
    with open(target, "wb" if overwrite else "xb") as handle:
        handle.write(content)
    return target


@dataclass(frozen=True)
class ToolOutput:
    """One file produced by a run, already downloaded from its signed link."""

    name: str
    """``image``, ``data`` (JSON) or ``file`` (another format, such as a PDF)."""
    url: str
    """Signed download link, valid until ``ToolResult.expires_at`` (one hour)."""
    content_type: str
    content: builtins.bytes = field(repr=False)
    key: str = ""
    """The tool output (see ``Tool.outputs``); ``out#0``, ``out#1``... in a batch."""
    bytes: Optional[int] = None
    """Size announced by the API, in bytes."""

    @property
    def extension(self) -> str:
        """File extension matching the content type, such as ``.png`` or ``.json``."""
        return extension_for(self.content_type)

    def json(self) -> Any:
        """The content parsed as JSON. Use it on the ``data`` output."""
        return json.loads(self.content)

    def save(self, path: Union[str, "os.PathLike[str]"], *, overwrite: bool = False) -> Path:
        """Write the content to ``path``. Refuses to replace a file unless ``overwrite=True``."""
        return _write(path, self.content, overwrite)


@dataclass(frozen=True)
class ToolResult:
    """What a run produced: ``image``, ``data``, and every file in ``outputs``."""

    module: str
    outputs: Tuple[ToolOutput, ...] = ()
    id: Optional[str] = None
    """Identifier of the run, useful when contacting support."""
    expires_at: Optional[str] = None
    """When the download links expire (ISO 8601). The files are deleted after 24 hours."""
    cost: Optional[int] = None
    """Credits charged for this run."""
    credits_balance: Optional[int] = None
    """Credits left on the account after this run."""
    message: Optional[str] = None
    """Optional note from the tool, such as a warning about the result."""

    def output(self, key_or_name: str) -> Optional[ToolOutput]:
        """The output with this key (such as ``out`` or ``out#1``), or else the first one
        with this name (``image``, ``data`` or ``file``). None when there is none."""
        for item in self.outputs:
            if item.key and item.key == key_or_name:
                return item
        return self._named(key_or_name)

    def _named(self, name: str) -> Optional[ToolOutput]:
        for item in self.outputs:
            if item.name == name:
                return item
        return None

    @property
    def file(self) -> Optional[ToolOutput]:
        """The produced file: the image, or else the first output that is not data."""
        image = self._named("image")
        if image is not None:
            return image
        for item in self.outputs:
            if item.name != "data":
                return item
        return None

    @property
    def image(self) -> Optional[bytes]:
        """Bytes of the produced file, or None when the tool only returns data."""
        produced = self.file
        return produced.content if produced is not None else None

    @property
    def content_type(self) -> Optional[str]:
        """Content type of the produced file, such as ``image/png``."""
        produced = self.file
        return produced.content_type if produced is not None else None

    @property
    def extension(self) -> str:
        """File extension of the produced file, such as ``.png``."""
        produced = self.file
        return produced.extension if produced is not None else ".json"

    @property
    def data(self) -> Any:
        """The ``data`` output parsed as JSON, or None when the tool returns none."""
        produced = self._named("data")
        return produced.json() if produced is not None else None

    def save(self, path: Union[str, "os.PathLike[str]"], *, overwrite: bool = False) -> Path:
        """Write the produced file. Refuses to replace a file unless ``overwrite=True``."""
        produced = self.file
        if produced is None:
            raise InvalidInputError(
                "This run produced no file, only data: read result.data instead."
            )
        return produced.save(path, overwrite=overwrite)

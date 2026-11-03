"""Input files and streamed request bodies."""

from __future__ import annotations

import asyncio
import io
import mimetypes
import os
import re
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import (
    IO,
    Any,
    AsyncGenerator,
    BinaryIO,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from urllib.parse import urlsplit

from .errors import InvalidInputError

DEFAULT_MAX_INPUT_BYTES = 3 * 1024 * 1024 * 1024
CHUNK_SIZE = 256 * 1024
SOURCE_URL_MAX_LENGTH = 2048

_UNSAFE_FILENAME_CHARS = re.compile(r'[\x00-\x1f\x7f"\\]')
_URL_FORBIDDEN_CHARS = re.compile(r"[\x00-\x20\x7f]")
_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/avif": ".avif",
    "application/json": ".json",
    "application/pdf": ".pdf",
}


@dataclass(frozen=True)
class SourceURL:
    """A public ``https`` URL to use as the tool's main input, instead of a local file::

        client.tools.run("removeBgApi", image=pixlub.SourceURL("https://example.com/a.jpg"))

    A plain string is always a local path. The URL must use https on port 443,
    carry no credentials and be at most 2048 characters long.
    """

    url: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", check_source_url(self.url))


FileSource = Union[str, "os.PathLike[str]", bytes, bytearray, memoryview, BinaryIO]
ImageSource = Union[FileSource, SourceURL]
InputValue = Union[ImageSource, List[ImageSource], Tuple[ImageSource, ...]]


def check_source_url(url: object) -> str:
    if not isinstance(url, str):
        raise InvalidInputError("SourceURL takes the address as a string.")
    url = url.strip()
    if not url or len(url) > SOURCE_URL_MAX_LENGTH:
        raise InvalidInputError(
            f"A source URL must be 1 to {SOURCE_URL_MAX_LENGTH} characters long."
        )
    if _URL_FORBIDDEN_CHARS.search(url):
        raise InvalidInputError("A source URL must not contain spaces or control characters.")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise InvalidInputError("Invalid source URL.") from None
    if parts.scheme != "https" or not parts.hostname:
        raise InvalidInputError("A source URL must start with https:// and name a host.")
    if port not in (None, 443):
        raise InvalidInputError("A source URL must use the standard https port (443).")
    if parts.username is not None or parts.password is not None:
        raise InvalidInputError("A source URL must not carry credentials.")
    return url


def extension_for(content_type: str) -> str:
    return _EXTENSIONS.get(content_type, ".img")


def sniff_content_type(data: bytes) -> Optional[str]:
    head = data[:16]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[:2] == b"BM":
        return "image/bmp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if head[4:8] == b"ftyp" and head[8:12] in (b"avif", b"avis"):
        return "image/avif"
    return None


class PreparedFile:
    def __init__(
        self,
        *,
        form_name: str,
        filename: str,
        content_type: str,
        size: int,
        data: Optional[memoryview] = None,
        path: Optional[Path] = None,
        stream: Optional[IO[bytes]] = None,
        start: int = 0,
        owns_stream: bool = False,
    ) -> None:
        self.form_name = form_name
        self.filename = filename
        self.content_type = content_type
        self.size = size
        self._data = data
        self._path = path
        self._stream = stream
        self._start = start
        self._owns_stream = owns_stream

    def __repr__(self) -> str:
        return (
            f"PreparedFile(form_name={self.form_name!r}, filename={self.filename!r}, "
            f"content_type={self.content_type!r}, size={self.size})"
        )

    def close(self) -> None:
        if self._owns_stream and self._stream is not None:
            self._stream.close()
        if self._data is not None:
            self._data.release()

    def chunks(self) -> Generator[bytes, None, None]:
        if self._data is not None:
            for offset in range(0, self.size, CHUNK_SIZE):
                yield bytes(self._data[offset : offset + CHUNK_SIZE])
            return
        handle = self._open()
        try:
            remaining = self.size
            while remaining:
                chunk = self._read(handle, remaining)
                remaining -= len(chunk)
                yield chunk
        finally:
            self._release(handle)

    async def achunks(self) -> AsyncGenerator[bytes, None]:
        if self._data is not None:
            for offset in range(0, self.size, CHUNK_SIZE):
                yield bytes(self._data[offset : offset + CHUNK_SIZE])
            return
        handle = await asyncio.to_thread(self._open)
        try:
            remaining = self.size
            while remaining:
                chunk = await asyncio.to_thread(self._read, handle, remaining)
                remaining -= len(chunk)
                yield chunk
        finally:
            self._release(handle)

    def _open(self) -> IO[bytes]:
        try:
            if self._path is not None:
                return self._path.open("rb")
            stream = self._stream
            if stream is None:
                raise InvalidInputError(f"Input '{self.form_name}' has nothing to read.")
            stream.seek(self._start)
            return stream
        except OSError as exc:
            raise InvalidInputError(
                f"Cannot read input '{self.form_name}': {exc.strerror or 'unknown error'}."
            ) from None

    def _read(self, handle: IO[bytes], remaining: int) -> bytes:
        try:
            chunk = handle.read(min(CHUNK_SIZE, remaining))
        except OSError as exc:
            raise InvalidInputError(
                f"Cannot read input '{self.form_name}': {exc.strerror or 'unknown error'}."
            ) from None
        if not chunk:
            raise InvalidInputError(
                f"Input '{self.form_name}' got shorter than its {self.size} bytes "
                "while it was being sent."
            )
        return bytes(chunk)

    def _release(self, handle: IO[bytes]) -> None:
        if self._path is not None:
            handle.close()


def prepare_file(source: FileSource, *, form_name: str, max_bytes: int) -> PreparedFile:
    label = f"Input '{form_name}'"
    if isinstance(source, SourceURL):
        raise InvalidInputError(f"{label}: a SourceURL can only fill the tool's main input.")
    if isinstance(source, (bytes, bytearray, memoryview)):
        try:
            data = memoryview(source).cast("B")
        except TypeError:
            raise TypeError("Buffers passed as an image must be contiguous bytes.") from None
        _check_size(label, data.nbytes, max_bytes, data.release)
        return _prepared(form_name, "", bytes(data[:16]), size=data.nbytes, data=data)
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        size, head = _measure_path(path, max_bytes)
        return _prepared(form_name, path.name, head, size=size, path=path)
    if hasattr(source, "read"):
        name = getattr(source, "name", None)
        filename = Path(name).name if isinstance(name, str) else ""
        return _prepare_stream(source, form_name, filename, max_bytes)
    raise TypeError(
        "An image must be a path, bytes, a binary file object or a pixlub.SourceURL, "
        f"not {type(source).__name__}."
    )


def _check_size(label: str, size: int, max_bytes: int, cleanup: Any = None) -> None:
    if size <= 0 or size > max_bytes:
        if cleanup is not None:
            cleanup()
        if size <= 0:
            raise InvalidInputError(f"{label} is empty.")
        raise InvalidInputError(f"{label} is {size} bytes; the limit is {max_bytes} bytes.")


def _prepared(form_name: str, filename: str, head: bytes, **content: Any) -> PreparedFile:
    content_type = (
        sniff_content_type(head) or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    )
    filename = _UNSAFE_FILENAME_CHARS.sub("_", filename).strip()
    if not filename:
        filename = f"{form_name}{extension_for(content_type)}"
    return PreparedFile(
        form_name=form_name, filename=filename, content_type=content_type, **content
    )


def _measure_path(path: Path, max_bytes: int) -> Tuple[int, bytes]:
    try:
        size = path.stat().st_size
        if not path.is_file():
            raise InvalidInputError(f"{path} is not a regular file.")
        if size == 0:
            raise InvalidInputError(f"{path} is empty.")
        if size > max_bytes:
            raise InvalidInputError(f"{path} is {size} bytes; the limit is {max_bytes} bytes.")
        with path.open("rb") as handle:
            head = handle.read(16)
    except OSError as exc:
        raise InvalidInputError(f"Cannot read {path}: {exc.strerror or 'unknown error'}.") from None
    return size, head


def _prepare_stream(
    stream: BinaryIO, form_name: str, filename: str, max_bytes: int
) -> PreparedFile:
    label = f"Input '{form_name}'"
    start = _seekable_start(stream)
    if start is not None:
        stream.seek(0, io.SEEK_END)
        size = stream.tell() - start
        stream.seek(start)
        _check_size(label, size, max_bytes)
        head = stream.read(16)
        stream.seek(start)
        if isinstance(head, str):
            raise TypeError("File objects must be opened in binary mode ('rb').")
        return _prepared(form_name, filename, bytes(head), size=size, stream=stream, start=start)

    spool = tempfile.TemporaryFile()
    try:
        size = 0
        while size <= max_bytes:
            chunk = stream.read(min(CHUNK_SIZE, max_bytes + 1 - size))
            if isinstance(chunk, str):
                raise TypeError("File objects must be opened in binary mode ('rb').")
            if not chunk:
                break
            spool.write(chunk)
            size += len(chunk)
        _check_size(label, size, max_bytes)
        spool.seek(0)
        head = spool.read(16)
    except BaseException:
        spool.close()
        raise
    return _prepared(form_name, filename, head, size=size, stream=spool, owns_stream=True)


def _seekable_start(stream: BinaryIO) -> Optional[int]:
    try:
        if callable(getattr(stream, "seekable", None)) and not stream.seekable():
            return None
        return stream.tell()
    except (OSError, AttributeError, ValueError):
        return None


class Body:
    content_type: str
    length: int

    def chunks(self) -> Generator[bytes, None, None]:
        raise NotImplementedError

    def achunks(self) -> AsyncGenerator[bytes, None]:
        raise NotImplementedError


class BytesBody(Body):
    def __init__(self, data: bytes, content_type: str) -> None:
        self._data = data
        self.content_type = content_type
        self.length = len(data)

    def chunks(self) -> Generator[bytes, None, None]:
        yield self._data

    async def achunks(self) -> AsyncGenerator[bytes, None]:
        yield self._data


class FileBody(Body):
    def __init__(self, file: PreparedFile) -> None:
        self._file = file
        self.content_type = file.content_type
        self.length = file.size

    def chunks(self) -> Generator[bytes, None, None]:
        return self._file.chunks()

    def achunks(self) -> AsyncGenerator[bytes, None]:
        return self._file.achunks()


class MultipartBody(Body):
    def __init__(self, fields: Sequence[Tuple[str, str]], files: Sequence[PreparedFile]) -> None:
        boundary = secrets.token_hex(16)
        self.content_type = f"multipart/form-data; boundary={boundary}"
        parts: List[Union[bytes, PreparedFile]] = []
        for name, value in fields:
            head = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            parts.append(head.encode() + value.encode() + b"\r\n")
        for file in files:
            head = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{file.form_name}"; '
                f'filename="{file.filename}"\r\n'
                f"Content-Type: {file.content_type}\r\n\r\n"
            )
            parts.extend((head.encode(), file, b"\r\n"))
        parts.append(f"--{boundary}--\r\n".encode())
        self._parts = parts
        self.length = sum(p.size if isinstance(p, PreparedFile) else len(p) for p in parts)

    def chunks(self) -> Generator[bytes, None, None]:
        for part in self._parts:
            if isinstance(part, PreparedFile):
                yield from part.chunks()
            else:
                yield part

    async def achunks(self) -> AsyncGenerator[bytes, None]:
        for part in self._parts:
            if isinstance(part, PreparedFile):
                async for chunk in part.achunks():
                    yield chunk
            else:
                yield part

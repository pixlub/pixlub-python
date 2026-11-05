# pixlub

Official Python SDK for the [Pixlub](https://pixlub.com) API. Python 3.9+, one
dependency: [httpx](https://www.python-httpx.org/).

Documentation: [getting started with the API](https://pixlub.com/en/docs/api/getting-started)
and [every module, its inputs and settings](https://pixlub.com/en/docs/modules).

<!-- GitHub plays the video; PyPI drops the <video> tag and shows the animated preview instead. -->
<video src="https://github.com/user-attachments/assets/b8ecd6c0-b65e-4c05-ae60-68e3d71c888a" controls muted width="100%">
  <a href="https://github.com/user-attachments/assets/b8ecd6c0-b65e-4c05-ae60-68e3d71c888a"><img src="https://raw.githubusercontent.com/pixlub/pixlub-python/main/docs/demo.webp" alt="Install pixlub, write a script, run a real request and get the cut-out"></a>
</video>

## Install

```bash
pip install pixlub
```

## Quick start

1. Create an API key in your Pixlub dashboard. API keys need a paid plan.
2. Put it in an environment variable, never in your source code:

   ```bash
   export PIXLUB_API_KEY="pix.xxxxxxxx.xxxxxxxx..."
   ```

3. Run a tool:

   ```python
   import pixlub

   client = pixlub.Client()
   result = client.tools.run("removeBgApi", image="photo.jpg")
   result.save(f"photo-nobg{result.extension}")
   ```

`result.extension` matches the produced file: `.webp` for `removeBgApi`, `.png`
for most tools. `save()` refuses to replace an existing file unless
`overwrite=True`.

## Results

```python
result = client.tools.run("faceBlurApi", image="photo.jpg", params={"mode": "auto"})

result.image            # bytes of the produced file (None if the tool only returns data)
result.data             # the data output, parsed JSON
result.cost             # credits charged
result.credits_balance  # credits left
result.output("out")    # an output by key, or by name ("image", "data", "file")
for output in result.outputs:
    print(output.key, output.name, output.content_type, len(output.content))
```

In a batch, output keys get the rank of the file: `out#0`, `out#1`… Each
`output.url` is a download link valid for one hour.

## Tools and settings

```python
for tool in client.tools.list(executable_only=True):
    print(tool.id, [i.id for i in tool.inputs], [o.key for o in tool.outputs])

schema = client.tools.schema("flip")  # JSON Schema of the settings
client.tools.run("flip", image="photo.jpg", params={"axis": "v"})
```

Invalid settings raise `InvalidRequestError`, with the reasons in `details`.

## Several files

A tool with several inputs takes `inputs=`, keyed by input id. An input that
takes several files takes a list:

```python
client.tools.run("mask", inputs={"image": "photo.png", "mask": "mask.png"})
client.tools.run("merge", inputs={"base": "bg.png", "overlay": ["1.png", "2.png"]})
client.tools.run("flip", image=["a.png", "b.png", "c.png"], params={"axis": "h"})  # 3 runs
```

When `tool.inputs[i].per_file` is true, each file is a separate run, charged
separately. A palette is passed in `params`, as a list of colors.

A file is a path, bytes, or a binary file object. To use a file from the web,
pass a `pixlub.SourceURL`; a plain string is always a local path.

```python
client.tools.run("removeBgApi", image=pixlub.SourceURL("https://example.com/photo.jpg"))
```

Large files are handled automatically: call `run()` the same way.

## Limits

| Limit | Value |
| --- | --- |
| File size, by plan | Free 10 MiB, Basic 50 MiB, Pro 3 GiB |
| File size, by tool | images 1 GiB, video and audio 3 GiB, Office 250 MiB |
| Image size | 40 megapixels |
| Files per run | 64 |
| Runs | 30 a minute per account, bursts of 10 |
| Download links | 1 hour |
| Results the SDK downloads | 1 GiB per file, 4 GiB per run (larger: use the link) |

## Async

`AsyncClient` has the same methods:

```python
import asyncio
import pixlub


async def main() -> None:
    async with pixlub.AsyncClient() as client:
        result = await client.tools.run("removeBgApi", image="photo.jpg")
        result.save(f"photo-nobg{result.extension}")


asyncio.run(main())
```

## Timeouts

`run()` waits for long runs until they finish, up to `job_timeout` (30 minutes
by default). After that, `JobTimeoutError` gives the `job_id`: the run may
still finish and may have been charged, so check it with `GET /v1/jobs/{job_id}`
rather than running the tool again.

```python
client = pixlub.Client(timeout=30, job_timeout=600)
result = client.tools.run("removeBgApi", image="photo.jpg", job_timeout=3600)
```

## Errors

API errors and refused arguments raise a `pixlub.PixlubError`. A wrong argument
type raises `TypeError`; a file that `save()` cannot write raises `OSError`.

| Exception | When |
| --- | --- |
| `ConfigurationError` | Missing or malformed key or base URL. |
| `InvalidInputError` | An argument was refused before anything was sent. |
| `UnknownToolError` | Unknown tool id. |
| `InvalidRequestError` | HTTP 400 or 422: invalid settings (`details`), `image_too_large` (`max_pixels`)… |
| `FileTooLargeError` | HTTP 413: file too large for the tool or your plan (`max_bytes`). |
| `AuthenticationError` | HTTP 401: unknown or revoked key. |
| `InsufficientCreditsError` | HTTP 402: not enough credits (`required_credits`). |
| `PermissionDeniedError` | HTTP 403: not included in your plan (`code`, `limit`). |
| `NotFoundError` | HTTP 404. |
| `RateLimitError` | HTTP 429: too many requests (`retry_after`). Never charged. |
| `ServerError`, `ServiceUnavailableError` | HTTP 5xx (`retry_after`). |
| `APIConnectionError`, `APITimeoutError` | No answer (`request_sent`). |
| `JobTimeoutError` | A run outlived `job_timeout` (`job_id`). |
| `UnexpectedResponseError` | The answer could not be read. |

Every `APIError` has `http_status`, `code` and `message`.

```python
try:
    result = client.tools.run("removeBgApi", image="photo.jpg")
except pixlub.InsufficientCreditsError as error:
    print("Top up your credits:", error.required_credits)
except pixlub.APIError as error:
    print(f"Refused ({error.http_status}): {error.message}")
```

## Retries and credits

The SDK retries a request only when it cannot cause a second charge, up to
`max_retries` times (2 by default, 0 to disable), and waits as long as the API
asks. Arguments are checked before anything is sent: a mistake costs nothing.
After a timeout or a server error on a run, check your balance before running
it again.

## Configuration

| Argument | Environment variable | Default |
| --- | --- | --- |
| `api_key` | `PIXLUB_API_KEY` | required |
| `base_url` | `PIXLUB_BASE_URL` | `https://api.pixlub.com` (`http://` only for localhost) |
| `timeout` | | 10 s connect, 180 s read, 60 s write |
| `job_timeout` | | 1800 s |
| `max_retries` | | 2 (0 to 5) |
| `max_input_bytes` | | 3 GiB |
| `http_client` | | a new `httpx.Client` (pass one for proxies or custom TLS) |

## Examples

[`examples/`](https://github.com/pixlub/pixlub-python/tree/main/examples):
`quickstart.py` for a first script, `batch_async.py` for many images at once.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy
```

Live tests run against a real API, and running a tool costs credits:

```bash
PIXLUB_BASE_URL=https://api.pixlub.com PIXLUB_API_KEY=... pytest tests/test_live.py
```

## License

MIT: see [LICENSE](https://github.com/pixlub/pixlub-python/blob/main/LICENSE).

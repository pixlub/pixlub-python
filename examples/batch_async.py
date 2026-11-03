"""Run one tool on many images, a few at a time.

    python examples/batch_async.py out/ photo1.jpg photo2.jpg photo3.jpg

At most CONCURRENCY runs at once. A failed image is reported and the others carry on.
"""

import asyncio
import sys
from pathlib import Path

import pixlub

TOOL = "removeBgApi"
CONCURRENCY = 4


async def process(
    client: pixlub.AsyncClient, gate: asyncio.Semaphore, index: int, source: Path, out_dir: Path
) -> str:
    async with gate:
        try:
            result = await client.tools.run(TOOL, image=source)
        except pixlub.PixlubError as error:
            return f"FAILED {source.name}: {error}"
        # The index keeps two inputs with the same file name from overwriting each other.
        target = out_dir / f"{index:03d}-{source.stem}-{TOOL}{result.extension}"
        result.save(target, overwrite=True)
        return f"ok {source.name} -> {target}"


async def main(out_dir: Path, sources: list) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(CONCURRENCY)
    async with pixlub.AsyncClient() as client:
        lines = await asyncio.gather(
            *(process(client, gate, i, Path(s), out_dir) for i, s in enumerate(sources, 1))
        )
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: python examples/batch_async.py <out-dir> <image> [<image> ...]")
    asyncio.run(main(Path(sys.argv[1]), sys.argv[2:]))

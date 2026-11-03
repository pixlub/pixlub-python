"""List the tools, then remove the background of an image.

export PIXLUB_API_KEY=pix....
python examples/quickstart.py photo.jpg
"""

import sys

import pixlub


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python examples/quickstart.py <image>")
        return 2

    client = pixlub.Client()
    tools = client.tools.list(executable_only=True)
    print(f"{len(tools)} tools available, for example: {', '.join(t.id for t in tools[:5])}")

    try:
        result = client.tools.run("removeBgApi", image=sys.argv[1])
    except pixlub.InsufficientCreditsError as error:
        print(f"Not enough credits: {error}")
        return 1
    except pixlub.APIError as error:
        print(f"The API refused the run: {error}")
        return 1

    output = result.save(f"output{result.extension}", overwrite=True)
    print(f"Saved {output} ({len(result.image)} bytes)")
    if result.credits_balance is not None:
        print(f"Credits left: {result.credits_balance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

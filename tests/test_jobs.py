from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from conftest import (
    IMAGE_PATH,
    JOB_ID,
    JOB_PATH,
    PNG,
    RUN_ID,
    accepted,
    job,
    json_response,
    succeeded,
)

import pixlub
from pixlub.errors import (
    APIConnectionError,
    InsufficientCreditsError,
    JobTimeoutError,
    NotFoundError,
    ServerError,
    UnexpectedResponseError,
)


def run_async(make_async_client, **kwargs: Any) -> Any:
    async def scenario() -> Any:
        async with make_async_client() as client:
            return await client.tools.run("removeBgApi", image=PNG, **kwargs)

    return asyncio.run(scenario())


def test_run_follows_a_long_job(api, client) -> None:
    api.queue("POST", "/v1/modules/removeBgApi", accepted())
    api.queue("GET", JOB_PATH, job("running"), succeeded())

    result = client.tools.run("removeBgApi", image=PNG)

    assert result.id == RUN_ID
    assert result.image == PNG
    assert result.outputs[0].key == "out"
    follow_ups = api.calls("GET", JOB_PATH)
    assert len(follow_ups) == 2
    assert follow_ups[0].headers["Prefer"] == "wait=25"
    assert follow_ups[0].headers["X-API-Key"]
    assert follow_ups[0].url.host == "api.pixlub.test"


def test_a_failed_job_raises_the_run_error(api, client) -> None:
    api.queue("POST", "/v1/modules/removeBgApi", accepted())
    api.queue(
        "GET",
        JOB_PATH,
        job(
            "failed",
            http_status=402,
            error="not enough credits",
            code="insufficient_credits",
            required=10,
        ),
    )

    with pytest.raises(InsufficientCreditsError) as caught:
        client.tools.run("removeBgApi", image=PNG)

    assert caught.value.http_status == 402
    assert caught.value.code == "insufficient_credits"
    assert caught.value.required_credits == 10
    assert caught.value.job_id == JOB_ID
    assert JOB_ID in str(caught.value)


def test_an_interrupted_job_is_a_server_error(api, client) -> None:
    api.queue("POST", "/v1/modules/removeBgApi", accepted())
    api.queue(
        "GET",
        JOB_PATH,
        job("failed", http_status=503, code="interrupted", error="interrupted"),
    )
    with pytest.raises(ServerError) as caught:
        client.tools.run("removeBgApi", image=PNG)
    assert caught.value.code == "interrupted"


def test_an_accepted_run_without_job_id_is_refused(api, client) -> None:
    api.queue("POST", "/v1/modules/removeBgApi", json_response({"status": "running"}, status=202))

    with pytest.raises(UnexpectedResponseError, match="job id"):
        client.tools.run("removeBgApi", image=PNG)


def test_follow_ups_pause_when_the_api_answers_at_once(api, client, waits) -> None:
    api.queue("POST", "/v1/modules/removeBgApi", accepted())
    api.queue("GET", JOB_PATH, job("queued"), job("running"), succeeded())

    client.tools.run("removeBgApi", image=PNG)

    assert waits.count(2.0) == 2


def test_follow_ups_do_not_pause_after_a_long_wait(api, client, clock, waits) -> None:
    def held_for_the_whole_wait(request: httpx.Request) -> httpx.Response:
        clock.now += 25
        return job("running")

    api.queue("POST", "/v1/modules/removeBgApi", accepted())
    api.queue("GET", JOB_PATH, held_for_the_whole_wait, succeeded())

    client.tools.run("removeBgApi", image=PNG)

    assert waits == []
    assert len(api.calls("GET", JOB_PATH)) == 2


def test_job_deadline_stops_following(api, make_client, waits) -> None:
    client = make_client(job_timeout=5)
    api.queue("POST", "/v1/modules/removeBgApi", accepted())
    api.queue("GET", JOB_PATH, *[job("running") for _ in range(10)])

    with pytest.raises(JobTimeoutError) as caught:
        client.tools.run("removeBgApi", image=PNG)

    error = caught.value
    assert error.job_id == JOB_ID
    assert JOB_ID in str(error) and "charged" in str(error)
    assert isinstance(error, APIConnectionError) and error.request_sent is True
    assert len(api.calls("GET", JOB_PATH)) == 3
    assert sum(waits) == pytest.approx(5.0)


def test_per_call_job_timeout_and_shorter_wait_near_the_deadline(api, client) -> None:
    api.queue("POST", "/v1/modules/removeBgApi", accepted())
    api.queue("GET", JOB_PATH, *[job("running") for _ in range(10)])

    with pytest.raises(JobTimeoutError):
        client.tools.run("removeBgApi", image=PNG, job_timeout=3)

    prefers = [r.headers["Prefer"] for r in api.calls("GET", JOB_PATH)]
    assert prefers == ["wait=3", "wait=1"]


@pytest.mark.parametrize("value", [0, -1, "30", True, float("inf")])
def test_invalid_job_timeout_is_rejected(make_client, value: Any) -> None:
    with pytest.raises(pixlub.ConfigurationError):
        make_client(job_timeout=value)


def test_follow_ups_are_read_for_at_least_35_seconds(api, client) -> None:
    api.queue("POST", "/v1/modules/removeBgApi", accepted())
    api.queue("GET", JOB_PATH, job("running"), succeeded())

    client.tools.run("removeBgApi", image=PNG, timeout=5)

    timeouts = [r.extensions["timeout"] for r in api.calls("GET", JOB_PATH)]
    assert all(t["read"] == 35.0 and t["connect"] == 5.0 for t in timeouts)
    assert api.calls("POST", "/v1/modules/removeBgApi")[0].extensions["timeout"]["read"] == 70.0


def test_a_lost_job_names_it_and_warns_of_the_charge(api, make_client) -> None:
    client = make_client(max_retries=0)
    api.queue("POST", "/v1/modules/removeBgApi", accepted())
    api.queue(
        "GET",
        JOB_PATH,
        json_response({"error": "Unknown job", "code": "job_not_found"}, status=404),
    )

    with pytest.raises(NotFoundError) as caught:
        client.tools.run("removeBgApi", image=PNG)

    assert caught.value.job_id == JOB_ID
    assert JOB_ID in str(caught.value)
    assert "may have been charged" in str(caught.value)


def test_a_follow_up_that_cannot_connect_names_the_job(api, make_client) -> None:
    client = make_client(max_retries=1)
    down = httpx.ConnectError("refused")
    api.queue("POST", "/v1/modules/removeBgApi", accepted())
    api.queue("GET", JOB_PATH, down, down)

    with pytest.raises(APIConnectionError) as caught:
        client.tools.run("removeBgApi", image=PNG)

    assert caught.value.job_id == JOB_ID
    assert "may have been charged" in str(caught.value)
    assert caught.value.__cause__ is None


def test_a_failed_download_after_a_job_names_the_job(api, make_client) -> None:
    client = make_client(max_retries=0)
    api.queue("POST", "/v1/modules/removeBgApi", accepted())
    api.queue("GET", JOB_PATH, succeeded())
    api.queue("GET", IMAGE_PATH, json_response({"error": "expired"}, status=403))

    with pytest.raises(pixlub.PermissionDeniedError) as caught:
        client.tools.run("removeBgApi", image=PNG)

    assert caught.value.job_id == JOB_ID
    assert JOB_ID in str(caught.value)


def test_a_failed_download_after_a_direct_run_says_it_was_charged(api, make_client) -> None:
    client = make_client(max_retries=0)
    api.queue("GET", IMAGE_PATH, json_response({"error": "gone"}, status=404))

    with pytest.raises(NotFoundError) as caught:
        client.tools.run("removeBgApi", image=PNG)

    assert caught.value.job_id is None
    assert RUN_ID in str(caught.value) and "charged" in str(caught.value)


def test_async_run_follows_a_long_job(api, make_async_client) -> None:
    api.queue("POST", "/v1/modules/removeBgApi", accepted())
    api.queue("GET", JOB_PATH, job("running"), succeeded())

    result = run_async(make_async_client)

    assert result.id == RUN_ID
    assert len(api.calls("GET", JOB_PATH)) == 2


def test_async_job_deadline_stops_following(api, make_async_client) -> None:
    api.queue("POST", "/v1/modules/removeBgApi", accepted())
    api.queue("GET", JOB_PATH, *[job("running") for _ in range(10)])

    with pytest.raises(JobTimeoutError) as caught:
        run_async(make_async_client, job_timeout=5)

    assert caught.value.job_id == JOB_ID
    assert len(api.calls("GET", JOB_PATH)) == 3


def test_async_follow_up_read_floor(api, make_async_client) -> None:
    api.queue("POST", "/v1/modules/removeBgApi", accepted())
    api.queue("GET", JOB_PATH, succeeded())

    run_async(make_async_client, timeout=5)

    assert api.calls("GET", JOB_PATH)[0].extensions["timeout"]["read"] == 35.0
    assert api.calls("POST", "/v1/modules/removeBgApi")[0].extensions["timeout"]["read"] == 70.0


def test_async_failed_job_and_lost_job_name_the_job(api, make_async_client) -> None:
    api.queue("POST", "/v1/modules/removeBgApi", accepted(), accepted())
    api.queue(
        "GET",
        JOB_PATH,
        job("failed", http_status=402, error="no credits", code="insufficient_credits"),
        json_response({"error": "Unknown job", "code": "job_not_found"}, status=404),
    )

    with pytest.raises(InsufficientCreditsError) as failed:
        run_async(make_async_client)
    with pytest.raises(NotFoundError) as lost:
        run_async(make_async_client)

    assert failed.value.job_id == JOB_ID
    assert lost.value.job_id == JOB_ID and "may have been charged" in str(lost.value)

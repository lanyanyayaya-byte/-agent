from concurrent.futures import ThreadPoolExecutor

from test_index_worker import configure, local_sidecar, make_pipeline, requests_at, until, wait_for_job
from test_library_management import import_file, library
from test_sidecar import SHUTDOWN, assert_clean_exit

DOCUMENTS = "/api/v1/knowledge/documents"
CONSENT = {"allow_cloud_embedding": True}


def index_path(book):
    return f"{DOCUMENTS}/{book.document_id}/index"


def jobs_path(book):
    return f"{DOCUMENTS}/{book.document_id}/versions/{book.document_version_id}/index-jobs"


def test_submission_consent_scope_and_retry_receipts(library, tmp_path):
    book = import_file(library, b"Geometry of a triangle.")
    other = import_file(library, b"Geology of basalt.")
    path = jobs_path(book)
    with local_sidecar(tmp_path) as (child, client, host, tokens):
        assert client.get(index_path(book), headers={ "Authorization": "Bearer " + tokens["host_token"] }).status_code == 403
        assert client.get(index_path(book), headers={"Authorization": ""}).status_code == 401
        assert client.get(index_path(book)).json() == {
            "document_id": book.document_id, "document_version_id": book.document_version_id,
            "job": None, "published_readiness": None,
        }
        for invalid in ({}, {"allow_cloud_embedding": False}, {"allow_cloud_embedding": "true"},
                        {**CONSENT, "lease_token": "fixture"}):
            assert client.post(path, json=invalid).status_code == 422
        wrong = path.replace(book.document_version_id, other.document_version_id)
        assert client.post(wrong, json=CONSENT).status_code == 409
        assert client.post(path.replace(book.document_version_id, "bogus"), json=CONSENT).status_code == 422

        with ThreadPoolExecutor(max_workers=6) as callers:
            responses = list(callers.map(lambda _: client.post(path, json=CONSENT), range(6)))
        assert {reply.status_code for reply in responses} == {200}
        job = responses[0].json()
        assert all(reply.json() == job for reply in responses)
        assert job["status"] == "queued"
        assert job["total_work"] is None and job["cancel_requested_at"] is None
        assert {"lease_token", "lease_expires_at", "instance_id", "published_readiness"}.isdisjoint(job)
        item = f"{path}/{job['job_id']}"
        lookup = client.get(item)
        assert lookup.headers["cache-control"] == "no-store" and lookup.json() == job
        assert client.get(f"{jobs_path(other)}/{job['job_id']}").status_code == 404
        assert client.post(f"{item}/retry", json=CONSENT).json()["error"]["code"] == "INDEX_JOB_NOT_TERMINAL"

        first = client.post(f"{item}/cancel").json()
        assert first["status"] == "cancelled"
        assert client.post(f"{item}/cancel").json() == first
        successor = client.post(f"{item}/retry", json=CONSENT).json()
        assert successor["job_id"] != job["job_id"] and successor["status"] == "queued"
        assert client.post(f"{item}/retry", json=CONSENT).json() == successor
        assert client.post(path, json=CONSENT).json() == successor

        next_book = import_file(library, b"Geometry of a plane triangle.", document_id=book.document_id)
        state = client.get(index_path(next_book)).json()
        assert state["document_version_id"] == next_book.document_version_id
        assert state["job"] is state["published_readiness"] is None
        assert client.post(path, json=CONSENT).status_code == 409
        assert client.post(f"{item}/retry", json=CONSENT).status_code == 409
        assert client.post(f"{DOCUMENTS}/{book.document_id}/delete").status_code == 204
        assert client.post(jobs_path(next_book), json=CONSENT).status_code == 404
        assert client.get(index_path(book)).status_code == 404
        assert client.post(SHUTDOWN, headers=host).status_code == 204
        assert_clean_exit(child, tokens)


def test_worker_progress_cancel_successor_and_published_capability(library, tmp_path):
    book = import_file(library, b"Geometry of an octagon.")
    pipeline = make_pipeline(library.store)
    with local_sidecar(tmp_path, "hold") as (child, client, host, tokens):
        configure(client, host)
        job = client.post(jobs_path(book), json=CONSENT).json()
        item = f"{jobs_path(book)}/{job['job_id']}"
        until(lambda: requests_at(tmp_path))
        state = client.get(index_path(book)).json()
        assert state["published_readiness"] == "ready_keyword"
        assert state["job"]["stage"] == "embedding" and state["job"]["status"] == "running"
        assert (state["job"]["completed_work"], state["job"]["total_work"]) == (0, 1)

        pending = client.post(f"{item}/cancel").json()
        assert pending["status"] == "running" and pending["cancel_requested_at"] is not None
        wait_for_job(pipeline, job["job_id"], "cancelled")
        assert client.get(index_path(book)).json()["published_readiness"] == "ready_keyword"

        (tmp_path / "release").touch()
        successor = client.post(f"{item}/retry", json=CONSENT).json()
        wait_for_job(pipeline, successor["job_id"], "succeeded")
        state = client.get(index_path(book)).json()
        assert state["published_readiness"] == "ready_hybrid"
        assert state["job"]["status"] == "succeeded"
        assert client.post(jobs_path(book), json=CONSENT).json() == state["job"]
        assert client.post(f"{jobs_path(book)}/{successor['job_id']}/retry", json=CONSENT).status_code == 409
        assert len(requests_at(tmp_path)) == 2
        assert client.post(SHUTDOWN, headers=host).status_code == 204
        assert_clean_exit(child, tokens)


def test_missing_key_and_version_update_keep_capability_scoped(library, tmp_path):
    book = import_file(library, b"Original octagon.")
    pipeline = make_pipeline(library.store)
    with local_sidecar(tmp_path) as (child, client, host, tokens):
        configure(client, host, key=False)
        job = client.post(jobs_path(book), json=CONSENT).json()
        wait_for_job(pipeline, job["job_id"], "failed")
        state = client.get(index_path(book)).json()
        assert state["published_readiness"] == "ready_keyword"
        assert state["job"]["error_code"] == "EMBEDDING_NOT_CONFIGURED"
        assert not requests_at(tmp_path)

        configure(client, host)
        item = f"{jobs_path(book)}/{job['job_id']}"
        successor = client.post(f"{item}/retry", json=CONSENT).json()
        wait_for_job(pipeline, successor["job_id"], "succeeded")
        assert client.get(index_path(book)).json()["published_readiness"] == "ready_hybrid"

        updated = import_file(library, b"New hexagon.", document_id=book.document_id)
        assert client.get(index_path(book)).json() == {
            "document_id": updated.document_id, "document_version_id": updated.document_version_id,
            "published_readiness": None, "job": None,
        }
        assert client.get(item).json()["error_code"] == "EMBEDDING_NOT_CONFIGURED"
        assert pipeline.published_readiness(pipeline.jobs.get_job(successor["job_id"])) == "ready_hybrid"
        assert requests_at(tmp_path) == ["Original octagon."]
        assert client.post(SHUTDOWN, headers=host).status_code == 204
        assert_clean_exit(child, tokens)

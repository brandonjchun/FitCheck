"""Constants shared by the load generator, its cleanup, and its tests.

**Separate from locustfile.py so that importing them does not import locust.**
Locust calls `gevent.monkey.patch_all()` at import, which rewrites `socket`,
`ssl`, and `threading` process-wide. That is correct for a load generator and
actively hazardous anywhere else: pulling it into the pytest process patches
SSL underneath httpx and anyio for every other test in the session, and gevent
itself warns that the result can be silently wrong rather than loudly broken.

So the cleanup script and the tests import this module, and only the load
generator imports locust.
"""

from pathlib import Path

# Where the queue-depth sampler writes, beside locust's own --csv output so one
# run produces one directory a reader can look at.
RESULTS = Path(__file__).resolve().parent / "results"

# A host that cannot resolve, by RFC 2606. The submit path is exercised in
# full -- validation, row, enqueue, worker pickup -- and the fetch fails at DNS
# without a packet leaving the machine.
#
# This is the single most important constant in the load test. Pointing it at a
# real board would make the p95 somebody else's server, invalidate the
# experiment, and aim a burst generator at a third party.
UNREACHABLE = "https://loadtest.invalid"

# Every account a run creates is prefixed so cleanup can find them without
# guessing, and so a stray one is recognisable months later.
EMAIL_PREFIX = "loadtest-"

PASSWORD = "loadtestPassword123"

# A minimal, valid, single-page PDF with a real text layer.
#
# Embedded as base64 rather than imported from `tests/conftest.py`, which is
# where the equivalent builder lives. Importing that module runs the test
# suite's isolation bootstrap, which repoints DATABASE_URL at `fitcheck_test` --
# so a load test that borrowed the builder would quietly measure the wrong
# database. Sixty lines of duplication avoided at the cost of one opaque
# constant is a bad trade; one opaque constant avoiding a wrong measurement is
# a good one.
#
# Every feed-reading user uploads this on start, because `/api/matches` and
# `/api/batches` are per-profile and 422 without one. Its extraction is queued
# like any other, which is realistic load rather than a side effect to hide.
RESUME_PDF_B64 = (
    "JVBERi0xLjQKMSAwIG9iago8PCAvVHlwZSAvQ2F0YWxvZyAvUGFnZXMgMiAwIFIgPj4KZW5kb2Jq"
    "CjIgMCBvYmoKPDwgL1R5cGUgL1BhZ2VzIC9LaWRzIFszIDAgUl0gL0NvdW50IDEgPj4KZW5kb2Jq"
    "CjMgMCBvYmoKPDwgL1R5cGUgL1BhZ2UgL1BhcmVudCAyIDAgUiAvTWVkaWFCb3ggWzAgMCA2MTIg"
    "NzkyXSAvUmVzb3VyY2VzIDw8IC9Gb250IDw8IC9GMSA1IDAgUiA+PiA+PiAvQ29udGVudHMgNCAw"
    "IFIgPj4KZW5kb2JqCjQgMCBvYmoKPDwgL0xlbmd0aCA1OCA+PgpzdHJlYW0KQlQgL0YxIDI0IFRm"
    "IDcyIDcwMCBUZCAoRW5naW5lZXIuIFB5dGhvbiBhbmQgUmVhY3QuKSBUaiBFVAplbmRzdHJlYW0K"
    "ZW5kb2JqCjUgMCBvYmoKPDwgL1R5cGUgL0ZvbnQgL1N1YnR5cGUgL1R5cGUxIC9CYXNlRm9udCAv"
    "SGVsdmV0aWNhID4+CmVuZG9iagp4cmVmCjAgNgowMDAwMDAwMDAwIDY1NTM1IGYgCjAwMDAwMDAw"
    "MDkgMDAwMDAgbiAKMDAwMDAwMDA1OCAwMDAwMCBuIAowMDAwMDAwMTE1IDAwMDAwIG4gCjAwMDAw"
    "MDAyNDEgMDAwMDAgbiAKMDAwMDAwMDM0OSAwMDAwMCBuIAp0cmFpbGVyCjw8IC9TaXplIDYgL1Jv"
    "b3QgMSAwIFIgPj4Kc3RhcnR4cmVmCjQxOQolJUVPRgo="
)

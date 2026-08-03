"""Parsing and normalizing lists of job-posting URLs.

A hand-collected URL list is the messiest input this system takes. It is
pasted from browser address bars, newsletters, and shared links, so it
arrives with tracking parameters, duplicate entries under different query
strings, blank lines, and the occasional pasted sentence.

Normalizing here rather than at the storage layer means the dedupe pass sees
canonical forms, and the count reported back to the user reflects what will
actually be fetched.
"""

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Matches a URI scheme prefix per RFC 3986. Used to tell "this line already
# declares a scheme" (http, mailto, javascript, file) from "this is a bare
# hostname" -- the two need opposite handling, and only the second gets an
# https:// prepended.
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")

# Query parameters that identify a marketing campaign rather than a document.
# Stripping them is what stops the same posting arriving twice because one
# copy came from a newsletter.
#
# Deliberately a denylist, not an allowlist. `?page=2` and `?gh_jid=12345`
# change which document you get -- an allowlist would have to enumerate every
# meaningful parameter on every job board, and dropping an unknown-but-real
# one silently fetches the wrong posting. Erring toward keeping a parameter
# costs a duplicate row; erring toward dropping it costs correctness.
TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "gclid",
        "fbclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "ref",
        "referer",
        "referrer",
        "source",
        "trk",
        "trackingid",
    }
)

ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def normalize_url(url: str) -> str | None:
    """Return a canonical form of `url`, or None if it is not usable.

    The transformations, and why each one is safe:

    - **Lowercase the scheme and host.** Both are case-insensitive per RFC
      3986. The *path* is not, and is deliberately left alone -- some boards
      serve case-sensitive slugs.
    - **Drop the fragment.** `#requirements` is a client-side scroll target
      that never reaches the server, so two URLs differing only there are one
      document.
    - **Strip tracking parameters** and sort what remains, so `?a=1&b=2` and
      `?b=2&a=1` agree.
    - **Remove a trailing slash** on non-root paths, since `/jobs/123` and
      `/jobs/123/` are the same posting everywhere in practice.
    - **Drop a default port**, so `:443` under https does not fork the key.

    Returns None for anything that is not an http(s) URL with a host --
    blank lines, pasted prose, `mailto:` links, and `file://` paths all land
    here. Rejecting non-http schemes matters beyond tidiness: these strings
    become fetch targets, and `file:///etc/passwd` reaching a worker is an
    SSRF primitive rather than a bad row.
    """
    candidate = url.strip()
    if not candidate:
        return None

    # Raw whitespace cannot appear in a URL, and its presence is the cheapest
    # reliable signal that a line is prose rather than a link. Without this
    # check the bare-host branch below turns "Here are the jobs I found:"
    # into "https://Here are the jobs I found:", which urlsplit accepts.
    if any(character.isspace() for character in candidate):
        return None

    if not _SCHEME_RE.match(candidate):
        # A bare "example.com/jobs/1" is a common paste, so assume https
        # rather than discarding it -- but only when the first segment looks
        # like a hostname. Requiring a dot is what separates that from
        # "just-words", which would otherwise become a fetch against a
        # nonexistent host.
        host_part = candidate.split("/", 1)[0].split("?", 1)[0]
        if "." not in host_part:
            return None
        candidate = f"https://{candidate}"

    try:
        parts = urlsplit(candidate)
    except ValueError:
        return None

    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        return None

    try:
        hostname, port = parts.hostname, parts.port
    except ValueError:
        # urlsplit defers parsing the port until the attribute is read, so a
        # malformed authority like "https://e.com:notaport/" raises here
        # rather than at split time.
        return None

    if not hostname:
        return None

    host = hostname.lower()
    if port and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"

    # keep_blank_values so `?debug` survives as a distinguishing parameter
    # rather than vanishing and merging two different pages.
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key.lower() not in {param.lower() for param in TRACKING_PARAMS}
        )
    )

    # "/" and "" name the same resource, so they collapse to one form --
    # otherwise "e.com" and "e.com/" become two rows for one posting.
    path = parts.path
    if path == "/":
        path = ""
    elif len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    return urlunsplit((scheme, host, path, query, ""))


def parse_url_list(text: str, cap: int) -> tuple[list[str], int, int]:
    """Parse an uploaded URL list into normalized, deduplicated URLs.

    Returns `(urls, rejected, duplicates)`:

    - `urls` -- up to `cap` normalized URLs, in the order they appeared. An
      LLM does not touch this, but people do: the order a user wrote their
      list in is meaningful to them and worth preserving in the progress view.
    - `rejected` -- lines that were not usable URLs.
    - `duplicates` -- lines that collapsed onto an earlier entry once
      normalized.

    All three counts go back in the response. A batch that silently ingests
    500 of someone's 4,000 lines is worse than one that refuses, because the
    user cannot tell which 3,500 are missing -- so the caller is told exactly
    what happened to every line it sent.

    The cap is applied *after* normalization and dedupe, so a list padded
    with the same posting under twenty tracking URLs is not penalised for it.
    """
    seen: set[str] = set()
    urls: list[str] = []
    rejected = 0
    duplicates = 0

    for line in text.splitlines():
        if not line.strip():
            # Blank lines are formatting, not errors. Counting them as
            # rejections would report a file with paragraph spacing as
            # half-broken.
            continue

        normalized = normalize_url(line)
        if normalized is None:
            rejected += 1
            continue

        if normalized in seen:
            duplicates += 1
            continue

        seen.add(normalized)
        if len(urls) < cap:
            urls.append(normalized)

    # Everything past the cap is reported as rejected, so accepted + rejected
    # + duplicates accounts for every non-blank line the user sent.
    over_cap = len(seen) - len(urls)

    return urls, rejected + over_cap, duplicates

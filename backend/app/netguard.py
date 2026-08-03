"""Refusing to fetch addresses that only this server can reach.

Every URL in this system arrives from a user and is then fetched by a worker
sitting inside our own network. That is the setup for SSRF: the attacker
cannot reach `169.254.169.254`, but our worker can, and it will fetch
whatever it is pointed at.

**This codebase has the worst-shaped version of it.** Most SSRF is *blind* --
the request happens but the response never comes back, so an attacker works
by inference. Ours stores the body as `job_postings.raw_text` and serves it
back to the submitter through the ordinary API, so a successful probe returns
its own loot in the response to a normal `GET`.

What is reachable from inside a deployed host and not from outside:

    169.254.169.254   Cloud instance metadata. AWS, GCP, Azure and
                      DigitalOcean all serve credentials from this address
                      with no authentication at all, because the service
                      trusts anything on the local link. This is the single
                      most exploited SSRF target in existence.
    127.0.0.1         Our own Redis -- which is the session store, so read
                      access to it is read access to every logged-in
                      account -- and Postgres.
    10/8, 172.16/12,
    192.168/16        Everything else on the private network: admin panels,
                      internal APIs, a colleague's laptop.

None of that is caught by the scheme validation in `urls.py`. They are all
ordinary `http://` URLs with ordinary hosts. robots.txt does not help either:
there is no robots.txt at a metadata endpoint, and an absent robots.txt
conventionally means *allow*.

**Two checks, deliberately split.** `blocked_literal_reason` judges an address
already written as an IP and does no I/O, so the request path can use it on
500 URLs without making 500 DNS queries. `assert_public_url` resolves and is
therefore the authoritative one -- it belongs in the worker. The split is why
this module imports nothing from the rest of the app.

**The resolving check is the one that actually holds**, and the literal check
is a courtesy that returns a better error sooner. Anything the literal check
misses -- a hostname, or one of the integer encodings of an address like
`http://2130706433/`, which is `127.0.0.1` written in decimal and which
`socket` resolves happily -- lands on the resolving check instead.
"""

import ipaddress
import re
import socket
from urllib.parse import urlsplit

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

# A host built only from numbers -- decimal, octal, or hex, dotted or not.
# Reached only after `ipaddress` has already declined to parse it, so a match
# means "looks like an address, is not a valid one".
_NUMERIC_HOST_RE = re.compile(r"^(0x[0-9a-f]+|[0-9]+)(\.(0x[0-9a-f]+|[0-9]+))*$")


class BlockedAddressError(Exception):
    """The URL resolves somewhere a user must not be able to send us."""


class UnresolvableHostError(BlockedAddressError):
    """DNS did not answer, so the address could not be judged either way.

    Separate from its parent because the two deserve opposite retry
    treatment. A blocked address is a property of the URL and will be blocked
    identically forever; a resolver that did not answer is very often a
    resolver that will answer in ten seconds. Collapsing them would
    dead-letter a legitimate job over a DNS hiccup, which is the expensive
    direction of the same mistake `providers/base.py` reasons about.
    """


def _unwrap(ip: IPAddress) -> IPAddress:
    """Reduce an IPv4-mapped IPv6 address to the IPv4 address inside it.

    `::ffff:127.0.0.1` is a real way to write loopback, and an `IPv6Address`
    holding it answers False to `is_loopback` -- the flag checks for `::1`.
    Unwrapping first means every predicate below sees the address that
    traffic will actually be delivered to.
    """
    mapped = getattr(ip, "ipv4_mapped", None)
    return mapped if mapped is not None else ip


def blocked_reason(ip: IPAddress) -> str | None:
    """Why `ip` must not be fetched, or None if it is a public address.

    The named checks come first because the message they produce is worth
    having in a log -- "link-local" tells you someone went looking for cloud
    credentials, where "not a global address" tells you nothing. The
    `is_global` backstop then catches the ranges not enumerated here, of
    which there are more than most lists remember: carrier-grade NAT at
    100.64/10, the TEST-NET blocks, 6to4 relay anycast.
    """
    ip = _unwrap(ip)

    if ip.is_loopback:
        return f"{ip} is a loopback address"
    if ip.is_link_local:
        # 169.254/16. The cloud metadata endpoint lives here, which is why
        # this case is worth naming separately from "private".
        return f"{ip} is link-local (cloud metadata lives at 169.254.169.254)"
    if ip.is_unspecified:
        # Ahead of is_private, which also claims 0.0.0.0 -- and "the
        # unspecified address" is the more useful half of the truth, since
        # 0.0.0.0 routes to localhost on most stacks.
        return f"{ip} is the unspecified address"
    if ip.is_private:
        return f"{ip} is a private address"
    if ip.is_reserved:
        return f"{ip} is reserved"
    if ip.is_multicast:
        return f"{ip} is multicast"
    if not ip.is_global:
        return f"{ip} is not a globally routable address"

    return None


# Names RFC 6761 reserves for loopback. Resolving them is guaranteed to come
# back to this machine, so they can be refused without asking anybody --
# and `metadata.google.internal` is included because GCP publishes it as the
# documented alias for 169.254.169.254 and it reads far more innocently.
_LOOPBACK_NAMES = frozenset({"localhost", "metadata.google.internal"})


def blocked_without_dns_reason(host: str) -> str | None:
    """Why `host` is unfetchable, using only checks that need no network.

    Covers literal IP addresses and the handful of names whose meaning is
    fixed by standard. Returns None for everything else -- which is **not an
    approval**. An ordinary hostname cannot be judged without resolving it,
    and that belongs in `assert_public_url`, away from the request path.

    Named for what it does rather than what it catches, because a function
    called `is_safe` would invite exactly the misreading that turns this into
    the only check anybody runs.
    """
    candidate = host.strip("[]").rstrip(".").lower()

    if candidate in _LOOPBACK_NAMES or candidate.endswith(".localhost"):
        return f"{candidate} is a reserved loopback name"

    try:
        return blocked_reason(ipaddress.ip_address(candidate))
    except ValueError:
        pass

    # Not a well-formed address, but it may still be an address. glibc's
    # resolver accepts encodings that `ipaddress` rejects: `2130706433` is
    # 127.0.0.1 as a decimal integer, `0177.0.0.1` is the same in octal, and
    # `0x7f.1` is another. Those are not curiosities -- they are the standard
    # way to write a blocked address so that a naive filter reads it as a
    # hostname.
    #
    # The rule that covers all of them without enumerating encodings: a host
    # made only of digits, dots, and hex prefixes is trying to be an address,
    # and one that will not parse as a valid public address does not get the
    # benefit of the doubt. No job board is reachable at an all-numeric name.
    if _NUMERIC_HOST_RE.match(candidate):
        return f"{candidate} is a non-standard encoding of an IP address"

    return None


def resolve_host(host: str, port: int) -> list[str]:
    """Every address `host` resolves to.

    Its own function so tests can substitute it without faking `socket`.
    """
    return [info[4][0] for info in socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)]


def assert_public_url(url: str) -> None:
    """Raise `BlockedAddressError` unless `url`'s host is safe to fetch.

    Every resolved address is checked, not just the first. A hostname
    answering with two A records -- one public, one `169.254.169.254` -- is a
    deliberate bypass, and which of the two a client ends up connecting to is
    not something this code decides.

    **The residual gap is DNS rebinding.** We resolve here, approve the
    result, and then httpx resolves again when it connects; a resolver
    controlled by the attacker can return a public address the first time and
    a private one the second. Closing it means pinning the connection to the
    address validated here via a custom transport, which is real work for a
    threat that needs attacker-controlled DNS. Recorded in
    PROJECTLIMITATIONS.md rather than built.
    """
    parts = urlsplit(url)
    host = parts.hostname
    if not host:
        raise BlockedAddressError(f"no host in {url!r}")

    # The no-DNS checks first, so a reserved name or a numeric-encoded
    # address is refused identically everywhere. Leaving it to resolution
    # would make the verdict depend on the local resolver: Windows declines
    # `2130706433` while glibc expands it to 127.0.0.1, and the workers run
    # on glibc while the tests often do not.
    reason = blocked_without_dns_reason(host)
    if reason is not None:
        raise BlockedAddressError(f"refusing to fetch {url}: {reason}")

    port = 443 if parts.scheme == "https" else 80
    try:
        port = parts.port or port
    except ValueError as exc:
        raise BlockedAddressError(f"malformed port in {url!r}") from exc

    try:
        addresses = resolve_host(host, port)
    except socket.gaierror as exc:
        # Not a block -- we could not find out. Reported as its own type so
        # the caller can retry rather than treating an unreachable resolver
        # as an attempted attack.
        raise UnresolvableHostError(f"could not resolve {host}: {exc}") from exc

    for address in addresses:
        reason = blocked_reason(ipaddress.ip_address(address))
        if reason is not None:
            raise BlockedAddressError(f"refusing to fetch {url}: {reason}")

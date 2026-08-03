"""The SSRF guard: which addresses this system refuses to fetch.

These are the tests for a security control, so they are written as the attack
rather than as the API. Each one names what an attacker gets if it fails,
because "blocked_reason returns a string" is not a claim anybody can check
against a threat.

The control exists because a worker inside our network will fetch any URL a
user submits, and the response is stored in `job_postings.raw_text` and served
back through the ordinary API. That last part is what makes this urgent rather
than theoretical: most SSRF is blind, and this one hands the body to whoever
asked for it.
"""

import ipaddress
import socket

import pytest

from app import netguard
from app.netguard import (
    BlockedAddressError,
    UnresolvableHostError,
    assert_public_url,
    blocked_without_dns_reason,
    blocked_reason,
)

# Routable, and belongs to nobody we will ever contact -- these tests never
# open a socket, but a public-looking address keeps that obvious.
PUBLIC = "93.184.216.34"


@pytest.fixture
def resolves_to(monkeypatch):
    """Aim hostname resolution wherever a test needs it."""

    def install(*addresses: str, error: str | None = None):
        def fake_resolve(host: str, port: int) -> list[str]:
            if error is not None:
                raise socket.gaierror(error)
            return list(addresses)

        monkeypatch.setattr(netguard, "resolve_host", fake_resolve)

    return install


class TestBlockedReason:
    @pytest.mark.parametrize(
        "address, why",
        [
            ("169.254.169.254", "the AWS/GCP/Azure metadata endpoint, which hands "
                                "out instance credentials to anything on the link"),
            ("127.0.0.1", "our own Redis session store and Postgres"),
            ("10.0.0.1", "the private network"),
            ("172.16.0.1", "the private network"),
            ("192.168.1.1", "the private network, including home routers"),
            ("0.0.0.0", "routes to localhost on most stacks"),
            ("::1", "loopback, spelled in IPv6"),
            ("224.0.0.1", "multicast"),
            ("100.64.0.1", "carrier-grade NAT -- the range people forget"),
        ],
    )
    def test_refuses(self, address: str, why: str) -> None:
        assert blocked_reason(ipaddress.ip_address(address)) is not None, why

    @pytest.mark.parametrize("address", [PUBLIC, "8.8.8.8", "2606:4700:4700::1111"])
    def test_allows_public_addresses(self, address: str) -> None:
        assert blocked_reason(ipaddress.ip_address(address)) is None

    @pytest.mark.parametrize(
        "address", ["::ffff:127.0.0.1", "::ffff:169.254.169.254", "::ffff:10.0.0.1"]
    )
    def test_ipv4_mapped_ipv6_is_unwrapped(self, address: str) -> None:
        """`::ffff:127.0.0.1` is loopback written as IPv6.

        An IPv6Address holding it answers False to `is_loopback`, because that
        flag looks for `::1`. Without unwrapping, this spelling walks straight
        through a check that blocks the same address written normally.
        """
        assert blocked_reason(ipaddress.ip_address(address)) is not None

    def test_names_the_metadata_endpoint_specifically(self) -> None:
        """A log line saying "link-local" tells an operator someone went
        looking for cloud credentials. "not global" tells them nothing."""
        reason = blocked_reason(ipaddress.ip_address("169.254.169.254"))

        assert "link-local" in reason
        assert "metadata" in reason


class TestBlockedWithoutDnsReason:
    def test_hostnames_are_not_judged(self) -> None:
        """Not an approval -- a hostname cannot be resolved without network
        I/O, which has no business running inside URL parsing. The verdict
        comes later, from assert_public_url."""
        assert blocked_without_dns_reason("boards.example.com") is None

    def test_bracketed_ipv6_is_understood(self) -> None:
        """urlsplit().hostname strips the brackets, but callers that pass a
        raw authority should not silently get a pass."""
        assert blocked_without_dns_reason("[::1]") is not None

    def test_literal_metadata_address_is_caught_without_dns(self) -> None:
        assert blocked_without_dns_reason("169.254.169.254") is not None

    @pytest.mark.parametrize(
        "host",
        [
            "localhost",
            "LOCALHOST",
            "localhost.",
            "db.localhost",
            "metadata.google.internal",
        ],
    )
    def test_reserved_loopback_names(self, host: str) -> None:
        """RFC 6761 fixes what `localhost` means, so no resolution is needed
        to know the answer. `metadata.google.internal` is GCP's documented
        alias for 169.254.169.254 and reads far more innocently than the
        address does.

        The trailing dot and the casing are both valid spellings that a
        naive equality check would let through.
        """
        assert blocked_without_dns_reason(host) is not None


class TestAssertPublicUrl:
    def test_public_host_is_allowed(self, resolves_to) -> None:
        resolves_to(PUBLIC)

        assert_public_url("https://boards.example.com/jobs/1")

    def test_hostname_pointing_at_metadata_is_blocked(self, resolves_to) -> None:
        """The attack the literal check cannot see.

        `evil.com` is a perfectly ordinary hostname; the A record is what
        points at the credentials. Nothing about the URL text gives it away.
        """
        resolves_to("169.254.169.254")

        with pytest.raises(BlockedAddressError, match="link-local"):
            assert_public_url("http://evil.example.com/")

    def test_every_resolved_address_is_checked(self, resolves_to) -> None:
        """Two A records, one public and one internal, is a deliberate bypass.

        Checking only the first address means the attacker controls which
        answer we validate by ordering their DNS response, and which one httpx
        connects to is a separate decision we do not make.
        """
        resolves_to(PUBLIC, "169.254.169.254")

        with pytest.raises(BlockedAddressError):
            assert_public_url("http://evil.example.com/")

    def test_offline_checks_run_before_resolution(self, resolves_to) -> None:
        """A reserved name must not depend on what the local resolver says.

        `metadata.google.internal` resolves to 169.254.169.254 on GCP and
        nowhere at all on a laptop. Deciding by resolution would make the
        verdict a property of the machine running the code -- blocked in
        production, "unresolvable, retry" in the tests that are supposed to
        prove it is blocked.
        """
        resolves_to(PUBLIC)

        with pytest.raises(BlockedAddressError, match="reserved"):
            assert_public_url("http://metadata.google.internal/")

    @pytest.mark.parametrize(
        "url",
        [
            "http://2130706433/",   # 127.0.0.1, decimal
            "http://0177.0.0.1/",   # 127.0.0.1, octal
            "http://0x7f.1/",       # 127.0.0.1, hex and packed
        ],
    )
    def test_numeric_encodings_are_refused(self, resolves_to, url: str) -> None:
        """The standard way to write a blocked address so a filter reads it
        as a hostname.

        `ipaddress` rejects all three, so a check that only tries to parse
        them lets them through -- and glibc's resolver expands every one to
        127.0.0.1. Relying on resolution to catch it is what makes the
        behaviour platform-dependent: Windows declines these, Linux does not,
        and the workers run on Linux.
        """
        resolves_to(PUBLIC)

        with pytest.raises(BlockedAddressError, match="non-standard"):
            assert_public_url(url)

    def test_ordinary_hostnames_are_not_caught_by_the_numeric_rule(
        self, resolves_to
    ) -> None:
        """The rule must not swallow real boards. Digits in a hostname are
        ordinary; a host made of *nothing but* digits is not."""
        resolves_to(PUBLIC)

        assert_public_url("https://boards3.example4.com/jobs/1")

    def test_unresolvable_host_is_its_own_error(self, resolves_to) -> None:
        """A resolver that did not answer is not an attack.

        The two get opposite retry treatment upstream: blocked is permanent,
        unresolvable is transient. Collapsing them dead-letters real work over
        a DNS hiccup.
        """
        resolves_to(error="Name or service not known")

        with pytest.raises(UnresolvableHostError):
            assert_public_url("https://nonexistent.example.com/")

    def test_unresolvable_is_still_a_blocked_address_error(self, resolves_to) -> None:
        """Subclassing means a caller that only knows the base type still
        fails closed rather than proceeding to fetch."""
        resolves_to(error="nope")

        with pytest.raises(BlockedAddressError):
            assert_public_url("https://nonexistent.example.com/")

    def test_url_with_no_host_is_rejected(self) -> None:
        with pytest.raises(BlockedAddressError):
            assert_public_url("file:///etc/passwd")

    def test_explicit_port_is_carried_into_resolution(self, monkeypatch) -> None:
        """Redis on 6379 and Postgres on 5432 are the interesting internal
        ports, so the port has to survive to the resolver rather than being
        defaulted away."""
        seen: list[int] = []

        def fake_resolve(host: str, port: int) -> list[str]:
            seen.append(port)
            return [PUBLIC]

        monkeypatch.setattr(netguard, "resolve_host", fake_resolve)

        assert_public_url("http://example.com:6379/")

        assert seen == [6379]

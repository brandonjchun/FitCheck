"""Seed the crawl targets.

    python -m scripts.seed_sources

Idempotent: upserts on (kind, board_token), so re-running adjusts display
names and intervals without duplicating a board or resetting its crawl
history. A script rather than a migration for the same reason the dev user
is -- migrations run in every environment, and which companies to crawl is a
choice, not a schema fact.

**Upsert-only. This never deletes.** Removing a line here does not remove the
row it created; the board simply stops being re-declared while continuing to
be crawled. `lever/plaid` is the standing example -- dropped from this list
once it was found to be empty, still sitting in `sources` afterwards. To
retire a board, disable or delete the row. Given `job_postings` reference
`sources`, disabling is the safer default.

**Every token below was verified live, and that is not ceremony.** A wrong
token does not raise. Greenhouse, Lever, and Ashby all answer `200` with an
empty collection for a board that does not exist, which reads downstream as
"a company that isn't hiring" rather than "a token that is wrong" -- so the
board sits in the ops dashboard looking healthy and contributing nothing,
forever. Of 172 candidates probed on 2026-08-04, **13 were bad**: 10 returned
404 and 3 returned `200` with zero postings. Those are excluded here:

    404        ashby/runwayml, greenhouse/doordash, greenhouse/matterport,
               greenhouse/wealthfront, greenhouse/zapier, lever/atlassian,
               lever/brex, lever/netflix, lever/quora, lever/sourcegraph
    silent 0   ashby/clerk, ashby/deel, lever/plaid

Several of those companies do run public boards -- on a *different* ATS than
guessed. `brex` and `wealthfront` are live on Greenhouse and dead on Lever;
`zapier` is live on Ashby and dead on Greenhouse. Guessing which ATS a company
uses is exactly the step that fails silently, which is why the counts in the
trailing comments are recorded: they are the evidence that a row was real at
seed time, and a board that later drops to zero is then visibly a change
rather than an unknown.

**Cost is no longer the reason to prefer one board kind -- with three
exceptions added at M11.** The original list was five boards, weighted to Lever
and Ashby because those return descriptions inline while Greenhouse required one
fetch per posting through a shared 1 rps bucket. Greenhouse now enumerates with
`?content=true` (see D-067), so Greenhouse, Lever, Ashby and Workable all cost
exactly one request per board per crawl regardless of size. That is what makes
108 Greenhouse boards tractable where 2 were not.

SmartRecruiters, Breezy and Rippling do not have that property: their listings
carry no descriptions, so each posting is a separate fetch. SmartRecruiters and
Breezy at least date their postings, which lets a re-crawl skip the unchanged
ones. Rippling dates nothing, so it re-fetches its whole board every time --
which is why it is seeded at a weekly interval rather than the 86400 default.
That interval is load-bearing, not a preference; see the entry.

**What running this actually commits you to.** 159 boards, ~19,170 postings.
Enumeration is trivial -- 159 requests. Extraction is not: every posting new
or changed is one LLM call, measured at ~10.5s against the local Ollama model,
and they serialize through a single instance. A cold fill is therefore on the
order of **55 hours** of continuous local inference. That is wall clock, not
money, and it drains without supervision -- but it is not a thing to start by
accident, and the boards are large enough that a few are worth thinking about
before the rest: `spacex` alone is 2,096 postings, more than 10% of the whole
catalog on its own.
"""

import logging

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db import SessionLocal
from app.models import Source

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("seed_sources")

# (kind, board_token, display_name, crawl_interval_seconds)
#
# Trailing comment is the posting count observed at seed time on 2026-08-04.
SOURCES = [
    # --- lever (15 boards, 535 postings) ---
    ("lever", "angellist", "AngelList", 86400),  # 21
    ("lever", "arcadia", "Arcadia", 86400),  # 11
    ("lever", "capital", "Capital", 86400),  # 41
    ("lever", "coupa", "Coupa", 86400),  # 92
    ("lever", "highspot", "Highspot", 86400),  # 19
    ("lever", "matchgroup", "Match Group", 86400),  # 82
    ("lever", "nium", "Nium", 86400),  # 39
    ("lever", "olo", "Olo", 86400),  # 9
    ("lever", "outreach", "Outreach", 86400),  # 32
    ("lever", "rover", "Rover", 86400),  # 22
    ("lever", "secureframe", "Secureframe", 86400),  # 15
    ("lever", "sonatype", "Sonatype", 86400),  # 25
    ("lever", "spotify", "Spotify", 86400),  # 103
    ("lever", "system1", "System1", 86400),  # 7
    ("lever", "wealthfront", "Wealthfront", 86400),  # 17
    # --- ashby (36 boards, 3,236 postings) ---
    ("ashby", "1password", "1Password", 86400),  # 68
    ("ashby", "airbyte", "Airbyte", 86400),  # 10
    ("ashby", "ashby", "Ashby", 86400),  # pre-existing seed row
    ("ashby", "benchling", "Benchling", 86400),  # 50
    ("ashby", "clickup", "ClickUp", 86400),  # 64
    ("ashby", "cohere", "Cohere", 86400),  # 141
    ("ashby", "confluent", "Confluent", 86400),  # 32
    ("ashby", "docker", "Docker", 86400),  # 58
    ("ashby", "drata", "Drata", 86400),  # 54
    ("ashby", "flock", "Flock", 86400),  # 5
    ("ashby", "fullstory", "FullStory", 86400),  # 1
    ("ashby", "headway", "Headway", 86400),  # 69
    ("ashby", "linear", "Linear", 86400),  # 24
    ("ashby", "miro", "Miro", 86400),  # 46
    ("ashby", "modal", "Modal", 86400),  # 31
    ("ashby", "moderntreasury", "Modern Treasury", 86400),  # 7
    ("ashby", "notion", "Notion", 86400),  # 111
    ("ashby", "openai", "OpenAI", 86400),  # 735
    ("ashby", "oyster", "Oyster", 86400),  # 20
    ("ashby", "patreon", "Patreon", 86400),  # 5
    ("ashby", "perplexity", "Perplexity", 86400),  # 86
    ("ashby", "plaid", "Plaid", 86400),  # 112 -- the *working* Plaid board
    ("ashby", "qumulo", "Qumulo", 86400),  # 28
    ("ashby", "ramp", "Ramp", 86400),  # 122
    ("ashby", "render", "Render", 86400),  # 35
    ("ashby", "replit", "Replit", 86400),  # 90
    ("ashby", "sanity", "Sanity", 86400),  # 25
    ("ashby", "sentry", "Sentry", 86400),  # 48
    ("ashby", "sierra", "Sierra", 86400),  # 174
    ("ashby", "snowflake", "Snowflake", 86400),  # 398
    ("ashby", "socure", "Socure", 86400),  # 93
    ("ashby", "strava", "Strava", 86400),  # 27
    ("ashby", "supabase", "Supabase", 86400),  # 57
    ("ashby", "telus-digital", "TELUS Digital", 86400),  # 121
    ("ashby", "vanta", "Vanta", 86400),  # 100
    ("ashby", "whoop", "WHOOP", 86400),  # 176
    ("ashby", "zapier", "Zapier", 86400),  # 13
    # --- greenhouse (108 boards, 15,399 postings) ---
    ("greenhouse", "adaptivebiotechnologies", "Adaptive Biotechnologies", 86400),  # 13
    ("greenhouse", "affirm", "Affirm", 86400),  # 181
    ("greenhouse", "airbnb", "Airbnb", 86400),  # 187
    ("greenhouse", "airtable", "Airtable", 86400),  # 40
    ("greenhouse", "amperity", "Amperity", 86400),  # 18
    ("greenhouse", "amplitude", "Amplitude", 86400),  # 41
    ("greenhouse", "anthropic", "Anthropic", 86400),  # 393
    ("greenhouse", "apolloio", "Apollo.io", 86400),  # 35
    ("greenhouse", "asana", "Asana", 86400),  # 144
    ("greenhouse", "betterment", "Betterment", 86400),  # 37
    ("greenhouse", "beyond", "Beyond", 86400),  # 5
    ("greenhouse", "bird", "Bird", 86400),  # 33
    ("greenhouse", "block", "Block", 86400),  # 205
    ("greenhouse", "brex", "Brex", 86400),  # 302 -- the *working* Brex board
    ("greenhouse", "calendly", "Calendly", 86400),  # 13
    ("greenhouse", "calm", "Calm", 86400),  # 1
    ("greenhouse", "carta", "Carta", 86400),  # 59
    ("greenhouse", "celonis", "Celonis", 86400),  # 249
    ("greenhouse", "chime", "Chime", 86400),  # 65
    ("greenhouse", "circleci", "CircleCI", 86400),  # 7
    ("greenhouse", "cloudflare", "Cloudflare", 86400),  # 289
    ("greenhouse", "coinbase", "Coinbase", 86400),  # 163
    ("greenhouse", "contentful", "Contentful", 86400),  # 27
    ("greenhouse", "coursera", "Coursera", 86400),  # 19
    ("greenhouse", "databricks", "Databricks", 86400),  # 806
    ("greenhouse", "datadog", "Datadog", 86400),  # 436
    ("greenhouse", "discord", "Discord", 86400),  # 44
    ("greenhouse", "dropbox", "Dropbox", 86400),  # 36
    ("greenhouse", "duolingo", "Duolingo", 86400),  # 66
    ("greenhouse", "elastic", "Elastic", 86400),  # 230
    ("greenhouse", "ensono", "Ensono", 86400),  # 66
    ("greenhouse", "epicgames", "Epic Games", 86400),  # 152
    ("greenhouse", "esri", "Esri", 86400),  # 441
    ("greenhouse", "faire", "Faire", 86400),  # 72
    ("greenhouse", "fandom", "Fandom", 86400),  # 6
    ("greenhouse", "fastly", "Fastly", 86400),  # 54
    ("greenhouse", "figma", "Figma", 86400),  # 177
    ("greenhouse", "fivetran", "Fivetran", 86400),  # 199
    ("greenhouse", "flex", "Flex", 86400),  # 48
    ("greenhouse", "general", "General", 86400),  # 1
    ("greenhouse", "gitlab", "GitLab", 86400),  # 185
    ("greenhouse", "glossgenius", "GlossGenius", 86400),  # 26
    ("greenhouse", "grafanalabs", "Grafana Labs", 86400),  # 140
    ("greenhouse", "gusto", "Gusto", 86400),  # 92
    ("greenhouse", "hightouch", "Hightouch", 86400),  # 70
    ("greenhouse", "honeycomb", "Honeycomb", 86400),  # 16
    ("greenhouse", "insomniac", "Insomniac Games", 86400),  # 2
    ("greenhouse", "instacart", "Instacart", 86400),  # 116
    ("greenhouse", "jetbrains", "JetBrains", 86400),  # 92
    ("greenhouse", "justworks", "Justworks", 86400),  # 93
    ("greenhouse", "karat", "Karat", 86400),  # 4
    ("greenhouse", "khanacademy", "Khan Academy", 86400),  # 22
    ("greenhouse", "lattice", "Lattice", 86400),  # 7
    ("greenhouse", "launchdarkly", "LaunchDarkly", 86400),  # 35
    ("greenhouse", "logicgate", "LogicGate", 86400),  # 12
    ("greenhouse", "lucidmotors", "Lucid Motors", 86400),  # 314
    ("greenhouse", "lyft", "Lyft", 86400),  # 161
    ("greenhouse", "make", "Make", 86400),  # 16
    ("greenhouse", "marqeta", "Marqeta", 86400),  # 44
    ("greenhouse", "mavenclinic", "Maven Clinic", 86400),  # 34
    ("greenhouse", "mercury", "Mercury", 86400),  # 55
    ("greenhouse", "mixpanel", "Mixpanel", 86400),  # 43
    ("greenhouse", "mongodb", "MongoDB", 86400),  # 402
    ("greenhouse", "monzo", "Monzo", 86400),  # 80
    ("greenhouse", "motive", "Motive", 86400),  # 13
    ("greenhouse", "naughtydog", "Naughty Dog", 86400),  # 17
    ("greenhouse", "netlify", "Netlify", 86400),  # 4
    ("greenhouse", "netskope", "Netskope", 86400),  # 141
    ("greenhouse", "nextdoor", "Nextdoor", 86400),  # 15
    ("greenhouse", "okta", "Okta", 86400),  # 344
    ("greenhouse", "orcasecurity", "Orca Security", 86400),  # 8
    ("greenhouse", "oscar", "Oscar Health", 86400),  # 261
    ("greenhouse", "oura", "Oura", 86400),  # 107
    ("greenhouse", "pagerduty", "PagerDuty", 86400),  # 20
    ("greenhouse", "peloton", "Peloton", 86400),  # 56
    ("greenhouse", "pendo", "Pendo", 86400),  # 36
    ("greenhouse", "pinterest", "Pinterest", 86400),  # 220
    ("greenhouse", "planetscale", "PlanetScale", 86400),  # 9
    ("greenhouse", "postman", "Postman", 86400),  # 106
    ("greenhouse", "reddit", "Reddit", 86400),  # 188
    ("greenhouse", "relativity", "Relativity Space", 86400),  # 363
    ("greenhouse", "remote", "Remote", 86400),  # 2
    ("greenhouse", "riotgames", "Riot Games", 86400),  # 163
    ("greenhouse", "robinhood", "Robinhood", 86400),  # 128
    ("greenhouse", "roblox", "Roblox", 86400),  # 221
    ("greenhouse", "rocketlab", "Rocket Lab", 86400),  # 395
    ("greenhouse", "samsara", "Samsara", 86400),  # 297
    ("greenhouse", "scaleai", "Scale AI", 86400),  # 216
    ("greenhouse", "seekout", "SeekOut", 86400),  # 4
    ("greenhouse", "smartsheet", "Smartsheet", 86400),  # 97
    ("greenhouse", "sofi", "SoFi", 86400),  # 65
    ("greenhouse", "sothebys", "Sotheby's", 86400),  # 60
    ("greenhouse", "spacex", "SpaceX", 86400),  # 2096 -- 11% of the catalog
    ("greenhouse", "squarespace", "Squarespace", 86400),  # 16
    ("greenhouse", "stripe", "Stripe", 86400),  # 546
    ("greenhouse", "sweetgreen", "Sweetgreen", 86400),  # 51
    ("greenhouse", "textio", "Textio", 86400),  # 1
    ("greenhouse", "thetradedesk", "The Trade Desk", 86400),  # 199
    ("greenhouse", "toast", "Toast", 86400),  # 294
    ("greenhouse", "twilio", "Twilio", 86400),  # 177
    ("greenhouse", "twitch", "Twitch", 86400),  # 62
    ("greenhouse", "udemy", "Udemy", 86400),  # 13
    ("greenhouse", "vercel", "Vercel", 86400),  # 81
    ("greenhouse", "verkada", "Verkada", 86400),  # 274
    ("greenhouse", "via", "Via", 86400),  # 175
    ("greenhouse", "webflow", "Webflow", 86400),  # 28
    ("greenhouse", "zoominfo", "ZoomInfo", 86400),  # 107
    ("greenhouse", "zscaler", "Zscaler", 86400),  # 302
    # --- workable (5 boards, 96 postings) ---
    #
    # Inline content, so these cost one request per crawl like Lever and Ashby.
    #
    # The skew is real and worth naming rather than hiding: Workable was founded
    # in Athens and its public boards are heavily Greek and European. That is a
    # coverage fact about this ATS, not a sampling mistake -- adding Workable
    # broadens the catalog geographically more than it does by industry.
    ("workable", "blueground", "Blueground", 86400),  # 28
    ("workable", "orfium", "Orfium", 86400),  # 21
    ("workable", "skroutz", "Skroutz", 86400),  # 10
    ("workable", "spotawheel", "Spotawheel", 86400),  # 34
    ("workable", "persado", "Persado", 86400),  # 3
    # --- smartrecruiters (1 board, 2 postings) ---
    #
    # Thin, and the reason is token discovery rather than the adapter. A
    # SmartRecruiters company id is not the brand slug: of Bosch, McDonalds,
    # Deloitte, IKEA, Publicis, Ubisoft, Sanofi, Vodafone and LinkedIn, every
    # one answered `200` with `totalFound: 0`. Only `Visa` resolved. Their ids
    # have to be read off a company's real careers URL, one at a time -- which
    # is the same silent-failure trap this file's header describes, at its
    # worst.
    ("smartrecruiters", "Visa", "Visa", 86400),  # 2
    # --- breezy (1 board, 3 postings) ---
    #
    # Per-posting fetch, but on a per-company subdomain, so two Breezy boards
    # do not contend for one token bucket the way two SmartRecruiters boards do.
    ("breezy", "breezy", "Breezy HR", 86400),  # 3
    # --- rippling (1 board, 738 postings) ---
    #
    # **Weekly, not daily, and the interval is the whole point of this entry.**
    # Rippling publishes no date of any kind in its listing, so
    # `_posting_needs_fetch` cannot skip anything and all 738 postings are
    # re-fetched on every crawl -- ~12 minutes through the shared
    # `ats.rippling.com` bucket. Daily, that is 84 minutes a week of fetching to
    # learn what a weekly crawl learns in 12.
    #
    # 604800 rather than 86400 for that reason alone. If a second Rippling board
    # is ever added, this is the number to copy.
    ("rippling", "rippling", "Rippling", 604800),  # 738
    # --- usajobs (not seeded) ---
    #
    # Deliberately absent, not forgotten. `enumerate_usajobs` is written and
    # tested but has never run against a live response, because the key is
    # per-developer and free from developer.usajobs.gov -- set USAJOBS_API_KEY
    # and USAJOBS_EMAIL, then add agencies by their `Organization` code:
    #
    #     ("usajobs", "TR", "Department of the Treasury", 86400),
    #     ("usajobs", "IN", "Department of the Interior", 86400),
    #
    # One source per agency rather than one for the whole corpus, so
    # `display_name` stays an employer and the company backfill keeps working.
    # Seeding one unverified is how a board ends up looking healthy and
    # contributing nothing, which is the failure this file exists to prevent.
]


def main() -> None:
    db = SessionLocal()
    try:
        for kind, token, name, interval in SOURCES:
            statement = (
                pg_insert(Source)
                .values(
                    kind=kind,
                    board_token=token,
                    display_name=name,
                    crawl_interval_seconds=interval,
                )
                .on_conflict_do_update(
                    index_elements=["kind", "board_token"],
                    # Deliberately does NOT touch `enabled`,
                    # `consecutive_failures`, or the crawl timestamps. Those
                    # are operational state: re-seeding after disabling a
                    # misbehaving board must not silently switch it back on.
                    set_={
                        "display_name": name,
                        "crawl_interval_seconds": interval,
                    },
                )
                .returning(Source.id)
            )
            source_id = db.execute(statement).scalar_one()
            logger.info("  %-11s %-26s -> source %s", kind, token, source_id)
        db.commit()
        total = db.query(Source).count()
        logger.info("%d sources declared, %d rows in `sources`", len(SOURCES), total)
    finally:
        db.close()


if __name__ == "__main__":
    main()

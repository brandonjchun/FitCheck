import { Link } from "react-router-dom";
import { HeroCanvas } from "../components/HeroCanvas";
import { Reveal } from "../components/Reveal";
import { useParallax } from "../hooks/useParallax";
import "./Landing.css";

/* --- Hero ------------------------------------------------------------- */

function Hero() {
  // Three depths, three speeds. The canvas trails furthest (it is the
  // "background"), the copy barely moves, and the floating cards lead the
  // scroll slightly so they read as nearest the viewer. That ordering is the
  // whole illusion -- get the signs wrong and it inverts into something that
  // feels subtly broken without being obviously so.
  const canvasLayer = useParallax<HTMLDivElement>({ speed: 90, zoom: 0.14 });
  const copyLayer = useParallax<HTMLDivElement>({ speed: 24, fade: 0.55 });
  const cardsLayer = useParallax<HTMLDivElement>({ speed: -46, drift: 10 });

  return (
    <section className="hero">
      <div className="hero-canvas-layer" ref={canvasLayer}>
        <HeroCanvas />
      </div>

      <div className="hero-grid" aria-hidden="true" />

      <div className="container hero-content">
        <div className="hero-copy" ref={copyLayer}>
          <span className="eyebrow">
            <span className="pulse-dot" aria-hidden="true" />
            Resume intelligence, explained
          </span>

          <h1>
            Know <span className="gradient-text">why</span> you fit — not just
            whether you do.
          </h1>

          <p className="hero-lede">
            FitCheck reads your resume and a job posting, derives a structured
            profile from each, and scores the pair on two independent signals.
            You get the breakdown, not a mystery percentage.
          </p>

          <div className="hero-cta">
            <Link to="/signin?mode=register" className="btn btn-primary btn-lg">
              Upload a resume
              <svg viewBox="0 0 20 20" width="17" height="17" fill="none" aria-hidden="true">
                <path
                  d="M4 10h12m0 0l-4.5-4.5M16 10l-4.5 4.5"
                  stroke="currentColor"
                  strokeWidth="1.9"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </Link>
            <a href="#how" className="btn btn-ghost btn-lg">
              See how it works
            </a>
          </div>

          <dl className="hero-stats">
            <div>
              <dt>Two signals</dt>
              <dd>Semantic + skill overlap</dd>
            </div>
            <div>
              <dt>Three buckets</dt>
              <dd>Matched · partial · missing</dd>
            </div>
            <div>
              <dt>Every score</dt>
              <dd>Traceable to evidence</dd>
            </div>
          </dl>
        </div>

        <div className="hero-cards" ref={cardsLayer} aria-hidden="true">
          <ScoreCard />
        </div>
      </div>

      <div className="hero-fade" aria-hidden="true" />
    </section>
  );
}

/** A static preview of the match breakdown the product actually renders. */
function ScoreCard() {
  return (
    <div className="score-card glass">
      <div className="score-card-head">
        <div>
          <p className="score-card-role">Senior Backend Engineer</p>
          <p className="score-card-co">Northwind Systems · Remote</p>
        </div>
        <div className="score-ring">
          <svg viewBox="0 0 44 44" width="52" height="52">
            <circle cx="22" cy="22" r="18" className="ring-track" />
            <circle cx="22" cy="22" r="18" className="ring-value" />
          </svg>
          <span>82</span>
        </div>
      </div>

      <div className="score-bars">
        <div className="score-bar">
          <span>Semantic</span>
          <div className="bar">
            <i style={{ width: "74%" }} />
          </div>
          <b>0.74</b>
        </div>
        <div className="score-bar">
          <span>Skills</span>
          <div className="bar">
            <i className="bar-alt" style={{ width: "87%" }} />
          </div>
          <b>0.87</b>
        </div>
      </div>

      <div className="score-skills">
        <span className="chip chip-ok">Python 5y</span>
        <span className="chip chip-ok">PostgreSQL</span>
        <span className="chip chip-ok">Redis</span>
        <span className="chip chip-partial">Kubernetes 1y</span>
        <span className="chip chip-missing">Terraform</span>
      </div>

      <p className="score-note">
        Partial: Kubernetes — has the skill, posting asks for 3+ years.
      </p>
    </div>
  );
}

/* --- How it works ----------------------------------------------------- */

const STEPS = [
  {
    n: "01",
    title: "Upload a resume",
    body: "PDF or DOCX. Text is extracted locally in milliseconds, then an LLM derives a structured profile — skills, years, seniority, education — validated against a strict schema.",
  },
  {
    n: "02",
    title: "Point at a posting",
    body: "Paste one URL, or upload a list of them. The API writes a row, enqueues the work, and returns 202 immediately. No request thread ever waits on a third-party fetch.",
  },
  {
    n: "03",
    title: "Read the breakdown",
    body: "Two independent scores and the skill groups behind them. Every extracted skill carries the verbatim phrase from your resume that justifies it.",
  },
];

function HowItWorks() {
  const glow = useParallax<HTMLDivElement>({ speed: 70, zoom: 0.2 });

  return (
    <section className="section how" id="how">
      <div className="section-glow" ref={glow} aria-hidden="true" />
      <div className="container">
        <Reveal>
          <span className="eyebrow">How it works</span>
          <h2 className="section-title">
            Three steps, one pipeline, zero&nbsp;guesswork.
          </h2>
        </Reveal>

        <ol className="steps">
          {STEPS.map((step, i) => (
            <Reveal as="li" key={step.n} delay={i * 110} className="step card">
              <span className="step-n">{step.n}</span>
              <h3>{step.title}</h3>
              <p>{step.body}</p>
            </Reveal>
          ))}
        </ol>
      </div>
    </section>
  );
}

/* --- Architecture ----------------------------------------------------- */

const LAYERS = [
  {
    title: "Async job queue",
    body: "Fetching an arbitrary URL takes 200ms to 30 seconds and fails often. That cannot live inside a request handler.",
  },
  {
    title: "Priority classes",
    body: "A crawl backlog and an interactive submission do not belong on the same queue. One user pasting a URL should not wait behind four thousand crawler jobs.",
  },
  {
    title: "Idempotent handlers",
    body: "Delivery is at-least-once, so every handler is written to converge: keyed upserts, early exit when the work is already done.",
  },
  {
    title: "Content-hash gating",
    body: "A repeat crawl re-sees mostly unchanged postings. If the text hashes the same, extraction and embedding are skipped entirely.",
  },
];

function Architecture() {
  const depth1 = useParallax<HTMLDivElement>({ speed: 52 });
  const depth2 = useParallax<HTMLDivElement>({ speed: -34, drift: -8 });

  return (
    <section className="section architecture" id="architecture">
      <div className="arch-orb arch-orb-a" ref={depth1} aria-hidden="true" />
      <div className="arch-orb arch-orb-b" ref={depth2} aria-hidden="true" />

      <div className="container">
        <Reveal>
          <span className="eyebrow">Under the hood</span>
          <h2 className="section-title">
            The interesting part isn't the score. It's everything that has to
            be true for the score to&nbsp;exist.
          </h2>
        </Reveal>

        <div className="layer-grid">
          {LAYERS.map((layer, i) => (
            <Reveal key={layer.title} delay={i * 90} className="layer card">
              <h3>{layer.title}</h3>
              <p>{layer.body}</p>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}

/* --- Scoring ---------------------------------------------------------- */

function Scoring() {
  const panel = useParallax<HTMLDivElement>({ speed: -22, zoom: 0.05 });

  return (
    <section className="section scoring" id="scoring">
      <div className="container scoring-inner">
        <Reveal className="scoring-copy">
          <span className="eyebrow">Explainable by construction</span>
          <h2 className="section-title">Two signals, because one isn't enough.</h2>
          <p>
            An embedding captures theme, and it is genuinely good at that. It is
            also perfectly capable of scoring a resume highly against a posting
            whose single mandatory requirement is missing.
          </p>
          <p>
            So skill overlap is computed explicitly and weighted, and the two
            are blended — with both sub-scores and the full breakdown always on
            screen. The weights are a stated judgment call, not a derived
            constant, and the interface says so.
          </p>
          <div className="formula">
            <code>final = 0.4 · semantic + 0.6 · skill</code>
          </div>
        </Reveal>

        <Reveal className="scoring-visual" delay={120}>
          <div className="scoring-panel card" ref={panel}>
            <div className="bucket bucket-ok">
              <h4>Matched</h4>
              <p>Required skill present, with enough years behind it.</p>
            </div>
            <div className="bucket bucket-partial">
              <h4>Partial</h4>
              <p>You have it. The posting wants more time with it.</p>
            </div>
            <div className="bucket bucket-missing">
              <h4>Missing</h4>
              <p>Required, and nothing in the resume supports it.</p>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}

/* --- Closing CTA ------------------------------------------------------ */

function ClosingCta() {
  const layer = useParallax<HTMLDivElement>({ speed: 40, zoom: 0.12 });

  return (
    <section className="section cta">
      <div className="cta-glow" ref={layer} aria-hidden="true" />
      <div className="container-narrow">
        <Reveal className="cta-inner">
          <h2 className="section-title">See what your resume actually says.</h2>
          <p>
            Upload it once. Every posting you check from then on is scored
            against the same structured profile.
          </p>
          <Link to="/signin?mode=register" className="btn btn-primary btn-lg">
            Get started
          </Link>
        </Reveal>
      </div>
    </section>
  );
}

export function Landing() {
  return (
    <main id="main">
      <Hero />
      <HowItWorks />
      <Architecture />
      <Scoring />
      <ClosingCta />
      <footer className="site-footer">
        <div className="container">
          <p>FitCheck — resume/JD matching with explainable scoring.</p>
          <p className="footer-muted">
            Built on FastAPI, PostgreSQL + pgvector, Redis + RQ, and React.
          </p>
        </div>
      </footer>
    </main>
  );
}

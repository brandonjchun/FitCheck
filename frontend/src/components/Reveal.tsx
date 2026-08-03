import { useEffect, useRef, type ReactNode } from "react";
import { useReducedMotion } from "../hooks/useReducedMotion";
import "./Reveal.css";

type RevealProps = {
  children: ReactNode;
  /** Stagger in ms, for revealing a row of cards in sequence. */
  delay?: number;
  /** Distance travelled on entry, in px. */
  distance?: number;
  className?: string;
  as?: "div" | "section" | "li" | "article";
};

/**
 * Reveals its children once, when they first scroll into view.
 *
 * Deliberately not wired to the parallax engine: this fires once and then
 * stops caring, so a per-element IntersectionObserver that disconnects on
 * first hit is cheaper than a subscription recomputed every frame forever.
 *
 * The starting state is applied by JS rather than CSS. If it were in the
 * stylesheet, a reduced-motion or no-JS visitor would be left with content
 * stuck at opacity 0 -- content hidden by a decoration that never runs.
 * Applying it from the effect means the default is always "visible".
 */
export function Reveal({
  children,
  delay = 0,
  distance = 26,
  className = "",
  as: Tag = "div",
}: RevealProps) {
  const ref = useRef<HTMLElement | null>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const el = ref.current;
    if (!el || reduced) return;

    el.style.opacity = "0";
    el.style.transform = `translate3d(0, ${distance}px, 0)`;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        el.style.transitionDelay = `${delay}ms`;
        el.classList.add("reveal-in");
        el.style.opacity = "";
        el.style.transform = "";
        observer.disconnect();
      },
      { threshold: 0.12, rootMargin: "0px 0px -8% 0px" },
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [delay, distance, reduced]);

  return (
    <Tag ref={ref as never} className={`reveal ${className}`}>
      {children}
    </Tag>
  );
}

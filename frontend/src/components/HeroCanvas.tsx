import { useEffect, useRef } from "react";
import { useReducedMotion } from "../hooks/useReducedMotion";
import "./HeroCanvas.css";

/**
 * The animated hero field.
 *
 * Generated rather than filmed. A video would be several megabytes, would
 * need a poster to avoid a flash, could not follow the theme, and would look
 * like stock footage of an office. This draws the thing the product actually
 * does: points drifting in a space, with links appearing between the ones
 * that are close together. That is a resume and a posting finding each other
 * in embedding space, which is the whole premise of the app.
 *
 * Three cost controls, because a full-viewport canvas is the easiest way to
 * ruin a page's battery and scroll performance:
 *
 *   1. Paused entirely when the tab is hidden or the canvas scrolls off
 *      screen. An unobserved animation is pure waste.
 *   2. Device pixel ratio is capped at 2. A 3x phone would otherwise ask for
 *      nine times the fill rate for a difference nobody can see on a blurred
 *      gradient.
 *   3. Node count scales with viewport area, so a phone draws a fraction of
 *      what a desktop does instead of the same load on weaker hardware.
 */

type Node = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
};

type Orb = {
  x: number;
  y: number;
  r: number;
  hue: string;
  phase: number;
  speed: number;
  driftX: number;
  driftY: number;
};

const LINK_DISTANCE = 148;
const MAX_DPR = 2;

function readColors(el: HTMLElement) {
  const styles = getComputedStyle(el);
  return {
    c1: styles.getPropertyValue("--hero-1").trim() || "#7c5cff",
    c2: styles.getPropertyValue("--hero-2").trim() || "#ff6b8a",
    c3: styles.getPropertyValue("--hero-3").trim() || "#38bdf8",
    ink: styles.getPropertyValue("--hero-ink").trim() || "#2a2140",
  };
}

export function HeroCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    let width = 0;
    let height = 0;
    let nodes: Node[] = [];
    let orbs: Orb[] = [];
    let rafId: number | null = null;
    let running = false;
    let colors = readColors(canvas);

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
      width = rect.width;
      height = rect.height;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      // Reset rather than accumulate: setTransform replaces the matrix, while
      // scale() would multiply it again on every resize.
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const density = Math.round((width * height) / 20000);
      const count = Math.max(18, Math.min(70, density));

      nodes = Array.from({ length: count }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.22,
        vy: (Math.random() - 0.5) * 0.22,
        r: Math.random() * 1.8 + 0.9,
      }));

      orbs = [
        { x: 0.22, y: 0.3, r: 0.42, hue: colors.c1, phase: 0, speed: 0.00013, driftX: 0.06, driftY: 0.05 },
        { x: 0.78, y: 0.34, r: 0.36, hue: colors.c2, phase: 2.1, speed: 0.00017, driftX: 0.05, driftY: 0.07 },
        { x: 0.5, y: 0.76, r: 0.46, hue: colors.c3, phase: 4.2, speed: 0.00011, driftX: 0.08, driftY: 0.04 },
      ];
    };

    const drawOrbs = (t: number) => {
      // Additive blending so overlaps brighten into a third colour instead of
      // one orb simply covering another.
      ctx.globalCompositeOperation = "lighter";
      for (const orb of orbs) {
        const cx = (orb.x + Math.sin(t * orb.speed + orb.phase) * orb.driftX) * width;
        const cy = (orb.y + Math.cos(t * orb.speed * 1.3 + orb.phase) * orb.driftY) * height;
        const radius = orb.r * Math.min(width, height);

        const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
        gradient.addColorStop(0, `${orb.hue}55`);
        gradient.addColorStop(0.45, `${orb.hue}22`);
        gradient.addColorStop(1, `${orb.hue}00`);

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalCompositeOperation = "source-over";
    };

    const drawNetwork = () => {
      // Links first so nodes sit on top of their own connections.
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i];
          const b = nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const distSq = dx * dx + dy * dy;
          if (distSq > LINK_DISTANCE * LINK_DISTANCE) continue;

          // Square root only for pairs that already passed the squared test.
          const dist = Math.sqrt(distSq);
          const strength = 1 - dist / LINK_DISTANCE;

          ctx.strokeStyle = colors.c1;
          ctx.globalAlpha = strength * 0.22;
          ctx.lineWidth = strength * 1.1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;

      for (const node of nodes) {
        ctx.fillStyle = colors.ink;
        ctx.globalAlpha = 0.32;
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    };

    const step = (dt: number) => {
      for (const node of nodes) {
        node.x += node.vx * dt;
        node.y += node.vy * dt;

        // Wrap rather than bounce. Bouncing makes edges visible as walls;
        // wrapping keeps the field reading as a slice of something larger.
        if (node.x < -20) node.x = width + 20;
        if (node.x > width + 20) node.x = -20;
        if (node.y < -20) node.y = height + 20;
        if (node.y > height + 20) node.y = -20;
      }
    };

    let lastTime = performance.now();

    const frame = (now: number) => {
      rafId = null;
      // Clamped so that returning to a backgrounded tab does not teleport
      // every node by however many seconds elapsed.
      const dt = Math.min(now - lastTime, 48) / 16.67;
      lastTime = now;

      ctx.clearRect(0, 0, width, height);
      drawOrbs(now);
      step(dt);
      drawNetwork();

      if (running) rafId = requestAnimationFrame(frame);
    };

    const renderStaticFrame = () => {
      ctx.clearRect(0, 0, width, height);
      drawOrbs(6000);
      drawNetwork();
    };

    const start = () => {
      if (running || reduced) return;
      running = true;
      lastTime = performance.now();
      rafId = requestAnimationFrame(frame);
    };

    const stop = () => {
      running = false;
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
    };

    resize();

    if (reduced) {
      // Still composed and still on-brand, just not moving. Rendering nothing
      // would leave a blank band where the design expects a field.
      renderStaticFrame();
    } else {
      start();
    }

    const resizeObserver = new ResizeObserver(() => {
      resize();
      if (reduced) renderStaticFrame();
    });
    resizeObserver.observe(canvas);

    const intersectionObserver = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) start();
        else stop();
      },
      { threshold: 0 },
    );
    intersectionObserver.observe(canvas);

    const onVisibility = () => {
      if (document.hidden) stop();
      else if (!reduced) start();
    };
    document.addEventListener("visibilitychange", onVisibility);

    // The theme toggle swaps the CSS variables this reads, but a canvas holds
    // no live link to them -- the colours were copied into JS at setup. This
    // re-reads them when the attribute flips so the field follows the theme.
    const themeObserver = new MutationObserver(() => {
      colors = readColors(canvas);
      for (let i = 0; i < orbs.length; i++) {
        orbs[i].hue = [colors.c1, colors.c2, colors.c3][i] ?? colors.c1;
      }
      if (reduced) renderStaticFrame();
    });
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    return () => {
      stop();
      resizeObserver.disconnect();
      intersectionObserver.disconnect();
      themeObserver.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [reduced]);

  return (
    <canvas
      ref={canvasRef}
      className="hero-canvas"
      // Decorative. The hero's meaning is carried entirely by the heading
      // beside it, so announcing this to a screen reader would add noise
      // without adding information.
      aria-hidden="true"
    />
  );
}

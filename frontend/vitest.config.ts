import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

/**
 * Test config, deliberately separate from vite.config.ts.
 *
 * Vitest reads this file *instead of* vite.config.ts when both exist, rather
 * than merging them -- which is why the react plugin is repeated here. The
 * duplication buys a build config that stays about building, with nothing
 * test-only leaking into what ships.
 */
export default defineConfig({
  plugins: [react()],
  test: {
    // jsdom, because everything under test touches the DOM: matchMedia,
    // localStorage, document.documentElement. A node environment would need
    // all three faked, and a fake matchMedia is exactly the thing most likely
    // to make a broken theme toggle look fine.
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    include: ["src/**/*.test.{ts,tsx}"],
  },
});

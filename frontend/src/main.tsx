import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import App from "./App.tsx";
import "./styles/global.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Two conservative defaults, both about not making the app feel jumpy.
      //
      // refetchOnWindowFocus is off because this app polls where polling is
      // warranted; refetching everything each time the user alt-tabs back is
      // load without information.
      //
      // retry is 1 rather than the default 3: a failing request here is
      // usually the backend being down, and three silent retries turn a
      // clear error into ten seconds of nothing.
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 30_000,
    },
  },
});

/* Theme is applied before React mounts.
 *
 * If this waited for useTheme's effect, a user who chose dark would get one
 * painted frame of light first -- the flash-of-wrong-theme every themed app
 * has to solve somewhere. Doing it here, synchronously, means the first
 * paint is already correct. It duplicates two lines from useTheme, which is
 * the right trade for removing a visible flash.
 */
const stored = localStorage.getItem("fitcheck-theme");
if (stored === "light" || stored === "dark") {
  document.documentElement.setAttribute("data-theme", stored);
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);

import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

/* Mocked at the module boundary so these tests are about the hooks' decisions
 * -- what a 401 means, what logout does to the cache -- rather than about
 * axios. The real client is exercised against the live API by the backend
 * suite; duplicating that here would test the mock. */
vi.mock("../api/client", () => ({
  authApi: {
    me: vi.fn(),
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
  },
}));

import { authApi } from "../api/client";
import { ME_KEY, useLogin, useLogout, useMe, useRegister } from "./useAuth";

/* Shaped to match UserResponse exactly. `created_at` is not decoration:
 * tsc typechecks these files even though vitest does not, so a partial
 * fixture fails the build rather than the test run. */
const USER = {
  id: 1,
  email: "dev@fitcheck.dev",
  created_at: "2026-08-03T00:00:00Z",
  is_admin: false,
};

/** An axios-shaped rejection, since that is what the hooks inspect. */
function httpError(status: number) {
  return Object.assign(new Error(`HTTP ${status}`), { response: { status } });
}

function harness() {
  // retry:false so a deliberate failure fails once instead of being retried
  // for the length of the test timeout.
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  return { queryClient, wrapper };
}

beforeEach(() => vi.clearAllMocks());

describe("useMe", () => {
  it("returns the user when the session is valid", async () => {
    vi.mocked(authApi.me).mockResolvedValue(USER);
    const { wrapper } = harness();

    const { result } = renderHook(() => useMe(), { wrapper });

    await waitFor(() => expect(result.current.data).toEqual(USER));
  });

  it("treats 401 as signed out, not as an error", async () => {
    /* The distinction the whole app leans on. A 401 here is the expected
     * answer for a visitor who has never signed in -- surfacing it as an
     * error state would put an error banner on the landing page for every
     * first-time visitor. */
    vi.mocked(authApi.me).mockRejectedValue(httpError(401));
    const { wrapper } = harness();

    const { result } = renderHook(() => useMe(), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.data).toBeNull();
    expect(result.current.isError).toBe(false);
  });

  it("still reports a real failure", async () => {
    /* The other half of the same decision. Swallowing every failure would
     * render a 500 or a dead backend as "signed out", which sends the user
     * to a sign-in form that cannot possibly work. */
    vi.mocked(authApi.me).mockRejectedValue(httpError(500));
    const { wrapper } = harness();

    const { result } = renderHook(() => useMe(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it("does not retry a 401", async () => {
    vi.mocked(authApi.me).mockRejectedValue(httpError(401));
    const { wrapper } = harness();

    const { result } = renderHook(() => useMe(), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(authApi.me).toHaveBeenCalledTimes(1);
  });
});

describe("useLogin", () => {
  it("seeds the cache instead of refetching", async () => {
    /* Login already returned the user. Invalidating would spend a second
     * round trip re-fetching a value we are holding, and show a loading
     * state immediately after a successful sign-in. */
    vi.mocked(authApi.login).mockResolvedValue(USER);
    const { queryClient, wrapper } = harness();

    const { result } = renderHook(() => useLogin(), { wrapper });
    await act(async () => {
      await result.current.mutateAsync({ email: USER.email, password: "pw" });
    });

    expect(queryClient.getQueryData(ME_KEY)).toEqual(USER);
    expect(authApi.me).not.toHaveBeenCalled();
  });

  it("leaves the cache alone when login fails", async () => {
    vi.mocked(authApi.login).mockRejectedValue(httpError(401));
    const { queryClient, wrapper } = harness();

    const { result } = renderHook(() => useLogin(), { wrapper });
    await act(async () => {
      await result.current
        .mutateAsync({ email: USER.email, password: "wrong" })
        .catch(() => {});
    });

    expect(queryClient.getQueryData(ME_KEY)).toBeUndefined();
  });
});

describe("useRegister", () => {
  it("signs the new account straight in", async () => {
    vi.mocked(authApi.register).mockResolvedValue(USER);
    const { queryClient, wrapper } = harness();

    const { result } = renderHook(() => useRegister(), { wrapper });
    await act(async () => {
      await result.current.mutateAsync({ email: USER.email, password: "pw" });
    });

    expect(queryClient.getQueryData(ME_KEY)).toEqual(USER);
  });
});

describe("useLogout", () => {
  it("empties the whole cache, not just the session", async () => {
    /* The one that matters on a shared machine. Everything cached was
     * fetched as somebody; leaving it means the next account watches the
     * previous one's profiles and job URLs flash on screen before their own
     * arrive. */
    vi.mocked(authApi.logout).mockResolvedValue(undefined);
    const { queryClient, wrapper } = harness();
    queryClient.setQueryData(ME_KEY, USER);
    queryClient.setQueryData(["profiles"], [{ id: 7, filename: "resume.pdf" }]);

    const { result } = renderHook(() => useLogout(), { wrapper });
    await act(async () => {
      await result.current.mutateAsync();
    });

    expect(queryClient.getQueryData(["profiles"])).toBeUndefined();
  });

  it("keeps the cache when the request fails", async () => {
    /* The session is still live on the server, so clearing here would show a
     * signed-out UI for a session that still works -- and the next request
     * would succeed, which is a confusing pair. */
    vi.mocked(authApi.logout).mockRejectedValue(httpError(500));
    const { queryClient, wrapper } = harness();
    queryClient.setQueryData(ME_KEY, USER);

    const { result } = renderHook(() => useLogout(), { wrapper });
    await act(async () => {
      await result.current.mutateAsync().catch(() => {});
    });

    expect(queryClient.getQueryData(ME_KEY)).toEqual(USER);
  });
});

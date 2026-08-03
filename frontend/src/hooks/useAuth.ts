import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { authApi, type User } from "../api/client";

export const ME_KEY = ["auth", "me"] as const;

/**
 * Current session, as a query rather than a context.
 *
 * Auth state is server state: the cookie lives in the browser, but whether it
 * is still *valid* is only knowable by asking. Mirroring it into a React
 * context means maintaining a copy that can disagree with the server -- and
 * it will, the first time a session is revoked in another tab.
 *
 * A 401 is the expected answer for a signed-out visitor, not an error, so it
 * is not retried and does not surface as a failure.
 */
export function useMe() {
  return useQuery<User | null>({
    queryKey: ME_KEY,
    queryFn: async () => {
      try {
        return await authApi.me();
      } catch (error) {
        const status = (error as { response?: { status?: number } })?.response?.status;
        if (status === 401) return null;
        throw error;
      }
    },
    retry: false,
    staleTime: 60_000,
  });
}

export function useLogin() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ email, password }: { email: string; password: string }) =>
      authApi.login(email, password),
    onSuccess: (user) => {
      // Seed rather than invalidate. Login already returned the user, so
      // refetching it immediately would be a redundant round trip against a
      // value we are holding.
      queryClient.setQueryData(ME_KEY, user);
    },
  });
}

export function useRegister() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ email, password }: { email: string; password: string }) =>
      authApi.register(email, password),
    onSuccess: (user) => queryClient.setQueryData(ME_KEY, user),
  });
}

export function useLogout() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: authApi.logout,
    onSuccess: () => {
      queryClient.setQueryData(ME_KEY, null);
      // Everything else in the cache was fetched as someone. Clearing it
      // stops the next account seeing the previous one's profiles flash on
      // screen before their own load.
      queryClient.clear();
    },
  });
}

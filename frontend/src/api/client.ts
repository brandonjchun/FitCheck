import axios, { AxiosError } from "axios";
import type { components } from "./schema";

/**
 * The single axios instance every request goes through.
 *
 * `withCredentials` is set here rather than per call, and it is the whole
 * reason auth works: the session is an HttpOnly cookie, so the browser will
 * only attach it cross-origin (5173 -> 8000) when the request opts in. The
 * frontend never sees, stores, or forwards a token -- that is the point of
 * the cookie design, and it is why an XSS on this page cannot read a session.
 */
export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? "http://localhost:8000",
  withCredentials: true,
});

/* --- Types, taken from the generated schema --------------------------- */

/* Aliased rather than redeclared. Hand-written duplicates of these drift the
 * moment the backend renames a field, and drift silently -- the compiler has
 * nothing to compare against. Sourcing them from the generated file turns a
 * backend rename into a build error here, which is the entire argument for
 * running openapi-typescript at all. */
export type Profile = components["schemas"]["ProfileUploadResponse"];
export type ProfileSummary = components["schemas"]["ProfileSummary"];
export type ExtractedSkill = components["schemas"]["ExtractedSkill"];
export type Job = components["schemas"]["JobResponse"];
export type User = components["schemas"]["UserResponse"];
export type JobStatus = Job["status"];

export type Batch = components["schemas"]["BatchStatusResponse"];
export type BatchCreated = components["schemas"]["BatchCreateResponse"];

export type Match = components["schemas"]["MatchResponse"];
export type MatchSkill = components["schemas"]["MatchSkill"];
export type SkillBucket = MatchSkill["bucket"];

export type OpsOverview = components["schemas"]["OpsOverview"];
export type QueueHealth = components["schemas"]["QueueHealth"];
export type SourceFreshness = components["schemas"]["SourceFreshness"];
export type Feedback = components["schemas"]["FeedbackResponse"];
export type FeedbackVerdict = components["schemas"]["FeedbackCreate"]["verdict"];
export type RecommendationRun = components["schemas"]["RecommendationRun"];
export type WorkerInfo = components["schemas"]["WorkerInfo"];
export type DeadLetterItem = components["schemas"]["DeadLetterItem"];

/**
 * Pull a human-usable message out of an axios error.
 *
 * FastAPI returns two different error shapes and they are easy to confuse:
 * `detail` is a string for a raised HTTPException, but an *array* of
 * per-field objects for a 422 validation failure. Rendering the array
 * directly is how a UI ends up showing "[object Object]" to a user who
 * mistyped a URL.
 */
export function errorMessage(error: unknown, fallback = "Something went wrong."): string {
  if (!(error instanceof AxiosError)) {
    return error instanceof Error ? error.message : fallback;
  }

  if (error.code === "ERR_NETWORK") {
    return "Cannot reach the API. Is the backend running on port 8000?";
  }

  const detail = error.response?.data?.detail;

  if (typeof detail === "string") return detail;

  if (Array.isArray(detail)) {
    const first = detail[0];
    if (first?.msg) {
      // `loc` is like ["body", "url"]; the first entry is the request part,
      // which is noise to a user. The rest is the field path.
      const field = Array.isArray(first.loc) ? first.loc.slice(1).join(".") : "";
      return field ? `${field}: ${first.msg}` : first.msg;
    }
  }

  return error.message || fallback;
}

/* --- Endpoints -------------------------------------------------------- */

export const authApi = {
  register: (email: string, password: string) =>
    api.post<User>("/api/auth/register", { email, password }).then((r) => r.data),

  login: (email: string, password: string) =>
    api.post<User>("/api/auth/login", { email, password }).then((r) => r.data),

  logout: () => api.post("/api/auth/logout").then(() => undefined),

  me: () => api.get<User>("/api/auth/me").then((r) => r.data),
};

export const profileApi = {
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    // Content-Type is deliberately not set. The browser has to generate it
    // so it can include the multipart boundary token; setting it by hand
    // produces a header with no boundary and a 422 that looks like a server
    // bug.
    return api.post<Profile>("/api/profiles", form).then((r) => r.data);
  },

  get: (id: number) => api.get<Profile>(`/api/profiles/${id}`).then((r) => r.data),

  /* Summaries, not full profiles: no raw_text and no skills. Enough to draw
   * the version list, and the open profile is fetched by id separately. */
  list: () => api.get<ProfileSummary[]>("/api/profiles").then((r) => r.data),

  activate: (id: number) =>
    api.post<Profile>(`/api/profiles/${id}/activate`).then((r) => r.data),

  remove: (id: number) => api.delete(`/api/profiles/${id}`).then(() => undefined),

  reextract: (id: number) =>
    api.post<Profile>(`/api/profiles/${id}/extract`).then((r) => r.data),
};

export const jobApi = {
  submit: (profileId: number, url: string) =>
    api
      .post<Job>("/api/jobs", { profile_id: profileId, url })
      .then((r) => r.data),

  get: (id: number) => api.get<Job>(`/api/jobs/${id}`).then((r) => r.data),

  list: (params: { profile_id?: number; status?: string; limit?: number } = {}) =>
    api.get<Job[]>("/api/jobs", { params }).then((r) => r.data),
};

export const batchApi = {
  /* profile_id rides in the query string, not the form body. The endpoint
   * declares it as a path-level parameter alongside the multipart file, so
   * appending it to the FormData would leave it unread and 422.
   *
   * A `File` becomes the `file` part and a `string` becomes the `urls` part.
   * The endpoint rejects both-at-once, so this sends exactly one -- the union
   * is what makes that unrepresentable here rather than merely unlikely. */
  create: (profileId: number, source: File | string) => {
    const form = new FormData();
    if (typeof source === "string") form.append("urls", source);
    else form.append("file", source);
    return api
      .post<BatchCreated>("/api/batches", form, { params: { profile_id: profileId } })
      .then((r) => r.data);
  },

  get: (id: number) => api.get<Batch>(`/api/batches/${id}`).then((r) => r.data),

  list: (limit = 20) =>
    api.get<Batch[]>("/api/batches", { params: { limit } }).then((r) => r.data),
};

/** Read-time feed filters. Every field optional; omitted means "don't care". */
export type FeedFilters = {
  origin?: "user_submission" | "recommendation";
  remote_only?: boolean;
  seniority?: string[];
  max_min_years?: number;
  include_closed?: boolean;
};

export const matchApi = {
  /* Keyed on the profile, not the job. A match is a (profile, posting) pair,
   * so the same posting scored against two of your resumes is two matches --
   * which is the point of keeping several versions around. */
  list: (profileId: number, limit = 25, filters: FeedFilters = {}) =>
    api
      .get<Match[]>("/api/matches", {
        params: { profile_id: profileId, limit, ...filters },
        /* Repeat the key per value -- `seniority=senior&seniority=staff` --
         * because that is what FastAPI's `list[str]` query parameter reads.
         * Axios would otherwise send `seniority[]=senior`, which arrives as a
         * differently-named parameter and is silently ignored. */
        paramsSerializer: { indexes: null },
      })
      .then((r) => r.data),

  get: (id: number) => api.get<Match>(`/api/matches/${id}`).then((r) => r.data),

  /* Append-only: every call records a new verdict rather than replacing the
   * last one, so interested-then-applied is preserved as a sequence. */
  feedback: (matchId: number, verdict: FeedbackVerdict) =>
    api
      .post<Feedback>(`/api/matches/${matchId}/feedback`, { verdict })
      .then((r) => r.data),

  /* Asks for the feed to be built. Safe to call whenever the feed looks
   * empty: the server answers `already_current` without touching the queue
   * if this profile already has a feed under the current scorer. */
  recommend: (profileId: number) =>
    api
      .post<RecommendationRun>("/api/matches/recommendations", null, {
        params: { profile_id: profileId },
      })
      .then((r) => r.data),
};

export const opsApi = {
  overview: () => api.get<OpsOverview>("/api/ops/overview").then((r) => r.data),

  deadLetter: (limit = 50) =>
    api
      .get<DeadLetterItem[]>("/api/ops/dead-letter", { params: { limit } })
      .then((r) => r.data),

  requeue: (jobId: number) =>
    api.post(`/api/ops/jobs/${jobId}/requeue`).then((r) => r.data),
};

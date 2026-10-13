// The gateway's API, relative to the page: the view is served at /dimos/blueprint_view, so "runs" is /dimos/runs.
// Anywhere else (vite's dev server) it goes through the dev proxy at /dimos/.
const BASE = location.pathname.startsWith("/dimos/") ? "" : "/dimos/";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

export async function getJson<T = unknown>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path.startsWith("../") ? path : BASE + path, options);
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(data?.error ?? `${response.status} ${response.statusText}`, response.status);
  }
  return data as T;
}

export const send = <T = unknown>(method: string, path: string, body?: unknown): Promise<T> =>
  getJson<T>(path, {
    method,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });

export const errorText = (error: unknown): string =>
  String((error as Error | null)?.message ?? error);

/** Framed by Desktop (its Details modal): it answers dimos:* messages and owns the close button. */
export const FRAMED = globalThis.parent !== undefined && globalThis.parent !== globalThis.window;

export const toDesktop = (message: Record<string, unknown>): void =>
  parent.postMessage(message, location.origin);

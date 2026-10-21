// The run this blueprint is in (GET /active's launch), and whether the saved config moved on since it started.
import { sameValue } from "dimos-react";
import type { Saved } from "./config.tsx";

export interface Launch {
  blueprint: string;
  phase: "starting" | "running" | "stopping" | "stopped" | "failed";
  pid: number;
  output?: string;
  runId: string | null;
  config?: { global?: Record<string, unknown>; modules?: Record<string, Record<string, unknown>> };
  steps?: { code: string; state: string; data?: { deployed?: number } }[];
}

/** A relaunch in flight: stopping the live run, then waiting for the new one. */
export type Relaunching = false | "stopping" | "starting";

const STEP_LABELS: Record<string, string> = {
  starting: "Starting dimOS",
  building: "Building the blueprint",
  starting_modules: "Starting modules",
  downloading_data: "Downloading the recording",
};

/** Whether the run started with other config than what's saved now (secrets, shown masked on both sides, aside). */
export function launchedWithOther(launch: Launch | null, saved: Saved | null): boolean {
  if (!launch?.config || !saved) {
    return false;
  }
  const masked = (value: unknown) => typeof value === "string" && /^•+$/.test(value);
  const differs = (
    a: Record<string, unknown>,
    b: Record<string, unknown>,
    base: Record<string, unknown> = {},
  ) =>
    [...new Set([...Object.keys(a), ...Object.keys(b)])].some((key) =>
      !masked(a[key]) && !masked(b[key]) &&
      !sameValue(key in a ? a[key] : base[key], key in b ? b[key] : base[key])
    );
  if (differs(saved.global, launch.config.global ?? {}, saved.defaults)) {
    return true;
  }
  const ran = launch.config.modules ?? {};
  const names = new Set([...Object.keys(saved.modules), ...Object.keys(ran)]);
  return [...names].some((name) => differs(saved.modules[name] ?? {}, ran[name] ?? {}));
}

function startingStep(
  launch: Launch | null,
  relaunching: Relaunching,
  relaunchedPid: number | null,
) {
  if (!launch || relaunching && launch.pid === relaunchedPid) {
    return STEP_LABELS.starting;
  }
  const download = (launch.output ?? "").split(/[\r\n]+/)
    .filter((line) => line.includes("Downloading LFS objects")).at(-1);
  if (download && !/\b100%|\bdone\b/i.test(download)) {
    return STEP_LABELS.downloading_data;
  }
  const step = (launch.steps ?? []).find((each) => each.state === "now");
  const deployed = step?.code === "starting_modules" ? step.data?.deployed : undefined;
  const label = STEP_LABELS[step?.code ?? ""] ?? STEP_LABELS.starting;
  return typeof deployed === "number" && deployed ? `${label} (${deployed} started)` : label;
}

export function relaunchLabel(
  launch: Launch | null,
  relaunching: Relaunching,
  relaunchedPid: number | null,
  stale: boolean,
): string {
  const phase = launch?.phase;
  if (
    relaunching === "stopping" && launch?.pid === relaunchedPid ||
    !relaunching && phase === "stopping"
  ) {
    return "Stopping…";
  }
  if (relaunching || phase === "starting") {
    return `${startingStep(launch, relaunching, relaunchedPid)}…`;
  }
  return stale ? "Relaunch to apply" : "Relaunch";
}

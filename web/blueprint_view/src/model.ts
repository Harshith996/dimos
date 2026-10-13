// A blueprint as GET /dimos/blueprints/{name} describes it, and how the view names and colors its parts.

export interface Stream {
  name: string;
  type: string;
  direction: string;
  wire?: string | null;
  topic?: string | null;
}

export interface Method {
  name: string;
  params: { name: string; type: string | null; default: string | null }[];
  return_type: string | null;
  doc: string;
}

export interface Module {
  name: string;
  class: string;
  doc?: string;
  summary?: string;
  streams: Stream[];
  rpcs?: Method[];
  skills?: Method[];
  file?: string | null;
  line?: number | null;
}

export interface Blueprint {
  name: string;
  file: string | null;
  line: number | null;
  modules: Module[];
}

export const reads = (stream: Stream): boolean => stream.direction !== "out";
export const writes = (stream: Stream): boolean => stream.direction !== "in";
export const className = (module: Module): string => module.class.split(".").pop() || module.name;
export const typeName = (type: string | null | undefined): string =>
  (type ?? "").split(".").pop() || type || "";
export const topicOf = (stream: Stream): string =>
  (stream.topic ?? stream.wire ?? stream.name).replace(/^\/+/, "");
/** The module's own name, when it isn't just its class lowercased. */
export const aliasOf = (module: Module): string | null =>
  module.name !== className(module).toLowerCase() ? module.name : null;

const TYPE_COLORS: Record<string, string> = {
  Image: "warn",
  CompressedImage: "warn",
  CameraInfo: "warn",
  PointCloud2: "violet",
  LaserScan: "violet",
  OccupancyGrid: "cat-1",
  Path: "cat-1",
  Odometry: "info",
  PoseStamped: "info",
  Twist: "info",
  TFMessage: "ok",
  Bool: "cat-2",
  String: "cat-2",
};
const PACKAGE_COLORS: Record<string, string> = {
  sensor_msgs: "warn",
  geometry_msgs: "info",
  nav_msgs: "cat-1",
  tf2_msgs: "ok",
  vision_msgs: "violet",
  std_msgs: "cat-2",
};
const PALETTE = ["info", "warn", "cat-1", "violet", "ok", "cat-2", "cat-3", "cat-4"];

/** The theme color a message type is drawn in: by type, else by package, else a stable pick. */
export function typeColor(type: string | null | undefined): string {
  const pkg = (type ?? "").split(".").find((part) => part.endsWith("_msgs")) ?? "";
  return TYPE_COLORS[typeName(type)] ?? PACKAGE_COLORS[pkg] ??
    PALETTE[[...(type ?? "")].reduce((sum, c) => sum + c.charCodeAt(0), 0) % PALETTE.length];
}

export const typeVar = (type: string | null | undefined): string => `var(--bv-${typeColor(type)})`;

const moduleKey = (name: string) => name.toLowerCase().replace(/[^a-z0-9]/g, "");

/** How many blueprints in the catalog use each module. */
export function moduleUsage(
  catalog: { blueprints?: { modules?: string[] }[] },
): Map<string, number> {
  const uses = new Map<string, number>();
  for (const blueprint of catalog.blueprints ?? []) {
    for (const key of new Set((blueprint.modules ?? []).map(moduleKey))) {
      uses.set(key, (uses.get(key) ?? 0) + 1);
    }
  }
  return uses;
}

/** Rarest first: what makes this blueprint different leads the list. */
export function byRarity(modules: Module[], uses: Map<string, number>): Module[] {
  const count = (module: Module) => uses.get(moduleKey(module.name)) ?? 0;
  return modules.map((module, index) => ({ module, index }))
    .sort((a, b) => count(a.module) - count(b.module) || a.index - b.index)
    .map(({ module }) => module);
}

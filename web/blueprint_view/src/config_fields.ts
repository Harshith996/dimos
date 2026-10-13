// The settings the config dialog edits: GlobalConfig's fields (from its JSON schema) and each module's config args,
// each with the kind of input it takes and the value it falls back to.

/** What the gateway sends in place of a saved secret. */
export const HIDDEN = "•••";

export const same = (a: unknown, b: unknown): boolean =>
  JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

export type Kind = "bool" | "enum" | "integer" | "number" | "string" | "path" | "json";

export interface FieldSpec {
  id: string;
  scope: "global" | "module";
  module?: string;
  name: string;
  kind: Kind;
  options?: unknown[];
  optionLabels?: string[];
  nullable: boolean;
  description: string | null;
  base: unknown;
  group: string;
  groupLabel?: string;
  secret: boolean;
  label?: string;
  placeholder?: string;
  docs?: string;
}

// deno-lint-ignore no-explicit-any
type Schema = Record<string, any>;

export interface GlobalConfig {
  schema?: Schema;
  defaults?: Record<string, unknown>;
  overrides?: Record<string, unknown>;
  secrets?: string[];
}

export interface ConfigArg {
  name: string;
  type?: string | null;
  default?: unknown;
  description?: string | null;
  base?: boolean;
  json_compatible?: boolean;
  secret?: boolean;
  choices?: unknown[];
  schema?: Schema;
  value?: unknown;
}

export interface BlueprintConfig {
  name?: string;
  modules: { module: string; class?: string; args: ConfigArg[]; error?: string }[];
  overrides?: Record<string, Record<string, unknown>>;
}

function resolveSchema(schema: Schema, root: Schema): Schema {
  if (schema.$ref) {
    const key = String(schema.$ref).split("/").pop()!;
    const target = root.$defs?.[key] ?? root.definitions?.[key];
    if (target) {
      return resolveSchema({ ...target, ...schema, $ref: undefined }, root);
    }
  }
  const variants: Schema[] | undefined = schema.anyOf ?? schema.oneOf;
  if (variants && !schema.type) {
    const nonNull = variants.filter((variant) => variant.type !== "null");
    if (nonNull.length === 1) {
      return {
        ...resolveSchema(nonNull[0], root),
        description: schema.description,
        default: schema.default,
      };
    }
  }
  return schema;
}

const nullableSchema = (schema: Schema): boolean =>
  ((schema.anyOf ?? schema.oneOf ?? []) as Schema[]).some((variant) => variant.type === "null") ||
  (Array.isArray(schema.type) && schema.type.includes("null"));

function kindOfSchema(schema: Schema): Kind {
  if (schema.enum) {
    return "enum";
  }
  const type = Array.isArray(schema.type)
    ? schema.type.find((name) => name !== "null")
    : schema.type;
  if (type === "boolean") {
    return "bool";
  }
  if (type === "integer" || type === "number") {
    return type;
  }
  if (type === "string") {
    return schema.format === "path" ? "path" : "string";
  }
  return "json";
}

function kindOfAnnotation(type: string | null | undefined): { kind: Kind; nullable: boolean } {
  const parts = (type ?? "").split("|").map((part) => part.trim()).filter(Boolean);
  const nullable = parts.includes("None") || /^Optional\[/.test(type ?? "");
  const main = parts.filter((part) => part !== "None");
  if (main.length !== 1) {
    return { kind: "json", nullable };
  }
  const one = main[0].replace(/^Optional\[(.*)\]$/, "$1");
  const kinds: Record<string, Kind> = {
    bool: "bool",
    int: "integer",
    float: "number",
    str: "string",
  };
  if (kinds[one]) {
    return { kind: kinds[one], nullable };
  }
  return { kind: /(^|\.)Path$/.test(one) ? "path" : "json", nullable };
}

function settable(arg: ConfigArg): boolean {
  if (arg.json_compatible !== undefined) {
    return arg.json_compatible;
  }
  const type = arg.type ?? "";
  return kindOfAnnotation(type).kind !== "json" || !!arg.choices ||
    /^(list|dict|tuple)\[(str|int|float|bool|,|\s)*\]$/.test(type);
}

/** Global keys sharing a first word (`robot_ip`, `robot_name`) group under it; the rest are "general". */
function globalGroups(keys: string[]): Map<string, string> {
  const word = (key: string) => key.toLowerCase().split("_")[0];
  const counts = new Map<string, number>();
  for (const key of keys) {
    counts.set(word(key), (counts.get(word(key)) ?? 0) + 1);
  }
  return new Map(keys.map((key) => [key, counts.get(word(key))! > 1 ? word(key) : "general"]));
}

export function globalFields(config: GlobalConfig): FieldSpec[] {
  const properties: Schema = config.schema?.properties ?? {};
  const keys = [...new Set([...Object.keys(properties), ...Object.keys(config.defaults ?? {})])];
  const groups = globalGroups(keys);
  const order = (key: string) =>
    `${groups.get(key) === "general" ? "" : groups.get(key)}\u0000${key.toLowerCase()}`;
  keys.sort((a, b) => order(a).localeCompare(order(b)));
  return keys.flatMap((key): FieldSpec[] => {
    const raw = properties[key] ?? {};
    const schema = resolveSchema(raw, config.schema ?? {});
    const kind = kindOfSchema(schema);
    if (kind === "json") {
      return [];
    }
    return [{
      id: key,
      scope: "global",
      name: key,
      kind,
      options: schema.enum,
      nullable: nullableSchema(raw),
      description: schema.description ?? null,
      base: config.defaults?.[key] ?? schema.default ?? null,
      group: groups.get(key) ?? "general",
      secret: config.secrets?.includes(key) ?? false,
    }];
  });
}

export function moduleFields(config: BlueprintConfig): FieldSpec[] {
  return config.modules.flatMap((module) =>
    module.args.filter((arg) => !arg.base && settable(arg)).map((arg): FieldSpec => {
      const fromSchema = arg.schema ? kindOfSchema(resolveSchema(arg.schema, arg.schema)) : null;
      const annotation = kindOfAnnotation(arg.type);
      return {
        id: `${module.module}.${arg.name}`,
        scope: "module",
        module: module.module,
        name: arg.name,
        kind: arg.choices ? "enum" : fromSchema ?? annotation.kind,
        options: arg.choices,
        nullable: arg.schema ? nullableSchema(arg.schema) : annotation.nullable,
        description: arg.description ?? null,
        base: "value" in arg ? arg.value : arg.default,
        group: module.module,
        groupLabel: module.class ? module.class.split(".").pop() : undefined,
        secret: arg.secret ?? false,
      };
    })
  );
}

export const asText = (value: unknown): string =>
  value === null || value === undefined
    ? ""
    : typeof value === "string"
    ? value
    : JSON.stringify(value);

export const valueText = (value: unknown): string =>
  value === null || value === undefined
    ? "—"
    : value === ""
    ? '""'
    : typeof value === "string"
    ? value
    : JSON.stringify(value);

/** The value typed text means for this field, or what it should have been. */
export function parseText(spec: FieldSpec, text: string): { value: unknown } | { error: string } {
  const trimmed = text.trim();
  if (trimmed === "" && spec.nullable) {
    return { value: null };
  }
  if (spec.kind === "integer" || spec.kind === "number") {
    const number = Number(trimmed);
    if (trimmed === "" || !Number.isFinite(number)) {
      return { error: "a number" };
    }
    if (spec.kind === "integer" && !Number.isInteger(number)) {
      return { error: "a whole number" };
    }
    return { value: number };
  }
  if (spec.kind === "json") {
    try {
      return { value: JSON.parse(text) };
    } catch {
      return { error: "JSON" };
    }
  }
  return { value: text };
}

import { describe, expect, it } from "vitest";
import { globalFields, moduleFields, parseText, same } from "./config_fields.ts";

describe("config fields", () => {
  it("reads GlobalConfig's schema into grouped inputs, skipping what can't be typed in", () => {
    const fields = globalFields({
      schema: {
        properties: {
          robot_ip: { anyOf: [{ type: "string" }, { type: "null" }] },
          robot_name: { type: "string" },
          viewer: { enum: ["rerun", "none"] },
          blob: { type: "object" },
        },
      },
      defaults: { viewer: "rerun" },
      secrets: [],
    });
    expect(fields.map((f) => [f.id, f.kind, f.group, f.nullable])).toEqual([
      ["robot_ip", "string", "robot", true],
      ["robot_name", "string", "robot", false],
      ["viewer", "enum", "general", false],
    ]);
  });

  it("takes a module's settable args, with the blueprint's value as the base", () => {
    const [field] = moduleFields({
      modules: [{
        module: "cam",
        class: "a.Camera",
        args: [{ name: "fps", type: "int", default: 5, value: 10 }, { name: "x", type: "Foo" }],
      }],
    });
    expect(field).toMatchObject({ id: "cam.fps", kind: "integer", base: 10, groupLabel: "Camera" });
  });

  it("parses typed text by kind", () => {
    const spec = { kind: "integer", nullable: true } as Parameters<typeof parseText>[0];
    expect(parseText(spec, "3")).toEqual({ value: 3 });
    expect(parseText(spec, "3.5")).toEqual({ error: "a whole number" });
    expect(parseText(spec, " ")).toEqual({ value: null });
    expect(same({ a: 1 }, { a: 1 })).toBe(true);
  });
});

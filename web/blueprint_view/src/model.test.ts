import { describe, expect, it } from "vitest";
import { byRarity, moduleUsage, topicOf, typeColor } from "./model.ts";

describe("model", () => {
  it("names topics without their leading slash, preferring the wired name", () => {
    expect(topicOf({ name: "image", type: "", direction: "out", topic: "/color" })).toBe("color");
    expect(topicOf({ name: "image", type: "", direction: "out", wire: "head" })).toBe("head");
    expect(topicOf({ name: "image", type: "", direction: "out" })).toBe("image");
  });

  it("colors by type, then package", () => {
    expect(typeColor("sensor_msgs.Image")).toBe("warn");
    expect(typeColor("geometry_msgs.Wrench")).toBe("info");
  });

  it("lists the rarest modules first", () => {
    const uses = moduleUsage({ blueprints: [{ modules: ["Lidar", "Map"] }, { modules: ["Map"] }] });
    const modules = ["map", "lidar", "solo"].map((name) => ({ name, class: name, streams: [] }));
    expect(byRarity(modules, uses).map((m) => m.name)).toEqual(["solo", "lidar", "map"]);
  });
});

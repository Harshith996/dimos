import { describe, expect, it } from "vitest";
import { exceptionsOf, isTraceback, parseTraceback, projectPath } from "./traceback.ts";

const PLAIN = `Traceback (most recent call last):
  File "/home/me/dimos/dimos/core/module.py", line 12, in start
    self.boot()
    ^^^^^^^^^^^
  File "/home/me/.venv/lib/python3.12/site-packages/lib.py", line 3, in boot
    raise ValueError("no")
ValueError: no
`;

describe("parseTraceback", () => {
  it("splits frames, code, markers and the exception, and tells library frames apart", () => {
    const items = parseTraceback(PLAIN);
    expect(items.map((item) => item.kind)).toEqual(["head", "frame", "frame", "exception"]);
    const [, own, library] = items;
    expect(own).toMatchObject({ kind: "frame", line: 12, func: "start", library: false });
    expect(own.kind === "frame" && own.code.map((c) => c.marker)).toEqual([false, true]);
    expect(library).toMatchObject({ kind: "frame", library: true });
    expect(exceptionsOf(items)).toEqual([{ depth: 0, text: "ValueError: no" }]);
  });

  it("reads paths from the dimos checkout's root", () => {
    expect(projectPath("/home/me/dimos/dimos/core/module.py")).toBe("dimos/core/module.py");
    expect(isTraceback(PLAIN)).toBe(true);
    expect(isTraceback("just text")).toBe(false);
  });
});

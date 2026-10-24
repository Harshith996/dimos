import { expect, test } from "vitest";
import { failedWhy, type Launch } from "./run.ts";

const launch = (fields: Partial<Launch>): Launch => ({
  blueprint: "nav",
  phase: "failed",
  pid: 1,
  runId: null,
  ...fields,
});

test("failedWhy: what the gateway said, when it said anything", () => {
  expect(failedWhy(launch({ error: "Error: Invalid blueprint configuration", output: "$ dimos run nav\n" })))
    .toBe("Error: Invalid blueprint configuration");
});

test("failedWhy: the last thing printed, when it didn't", () => {
  expect(failedWhy(launch({ output: "$ dimos run nav\nStarting DimOS\nKilled\n\n" }))).toBe("Killed");
});

test("failedWhy: a launch that said and printed nothing still reads as a sentence", () => {
  expect(failedWhy(launch({}))).toBe("it printed nothing");
  expect(failedWhy(launch({ error: "   ", output: "  \n \n" }))).toBe("it printed nothing");
  expect(failedWhy(null)).toBe("it printed nothing");
});

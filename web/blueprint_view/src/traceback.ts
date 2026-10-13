// Python tracebacks (plain, chained and exception groups) split into frames, code, markers and exceptions.
const FRAME = /^(\s*)File "(.+)", line (\d+)(?:, in (.*))?$/;
const MARKER = /^\s*[\^~]+[\^~\s]*$/;
const HEAD =
  /^(?:Exception Group )?Traceback \(most recent call last\):|^During handling of the above exception|^The above exception was the direct cause/;
const LIBRARY =
  /\/site-packages\/|\/dist-packages\/|\/\.venv\/|\/lib\/python\d[\d.]*\/|\/uv\/python\/|\/nix\/store\/|^<frozen |^<string>/;

export type TracebackItem =
  | {
    kind: "frame";
    gutter: string;
    indent: string;
    file: string;
    line: number;
    func: string;
    library: boolean;
    code: { text: string; marker: boolean }[];
  }
  | { kind: "head" | "text" | "sep"; gutter: string; text: string }
  | { kind: "exception"; gutter: string; depth: number; lines: string[] };

export const isTraceback = (text: unknown): text is string =>
  typeof text === "string" && /Traceback \(most recent call last\):/.test(text);

/** A path inside the dimos checkout, from the repo root (what the editor and /dimos/source take). */
export function projectPath(file: string): string {
  const at = file.lastIndexOf("/dimos/");
  return at >= 0 ? file.slice(at + 1) : file;
}

const depthOf = (gutter: string) => Math.max(0, Math.round((gutter.search(/[|+]/) - 2) / 2));

export function parseTraceback(text: string): TracebackItem[] {
  const grouped = /^\s*\+ Exception Group Traceback/m.test(text);
  const items: TracebackItem[] = [];
  const last = () => items[items.length - 1];
  for (const raw of text.replace(/\n$/, "").split("\n")) {
    const gutterMatch = grouped ? /^(\s*[|+](?: |$))?(.*)$/.exec(raw) : null;
    const gutter = gutterMatch?.[1] ?? "";
    const body = gutterMatch ? gutterMatch[2] : raw;
    if (grouped && /^\s*\+-[-+]*(\s.*)?$/.test(raw) && /-{4}/.test(raw)) {
      items.push({ kind: "sep", gutter: "", text: raw });
      continue;
    }
    const frame = FRAME.exec(body);
    const previous = last();
    if (frame) {
      items.push({
        kind: "frame",
        gutter,
        indent: frame[1],
        file: frame[2],
        line: Number(frame[3]),
        func: frame[4] ?? "",
        library: LIBRARY.test(frame[2]),
        code: [],
      });
    } else if (HEAD.test(body)) {
      items.push({ kind: "head", gutter, text: body });
    } else if (previous?.kind === "frame" && /^\s/.test(body) && body.trim()) {
      previous.code.push({ text: body, marker: MARKER.test(body) });
    } else if (!body.trim()) {
      items.push({ kind: "text", gutter, text: body });
    } else if (!/^\s/.test(body) && previous?.kind === "exception" && previous.gutter === gutter) {
      previous.lines.push(body);
    } else if (!/^\s/.test(body)) {
      items.push({ kind: "exception", gutter, depth: depthOf(gutter), lines: [body] });
    } else {
      items.push({ kind: "text", gutter, text: body });
    }
  }
  return items;
}

/** What was raised, outermost first, with its nesting in an exception group. */
export const exceptionsOf = (items: TracebackItem[]): { depth: number; text: string }[] =>
  items.flatMap((item) =>
    item.kind === "exception" ? [{ depth: item.depth, text: item.lines.join("\n") }] : []
  );

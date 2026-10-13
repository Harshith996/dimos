// A run's structured log (GET /dimos/runs/{id}/log), followed live, with tracebacks drawn frame by frame.
import { Checkbox, Modal, SearchInput, Select } from "dimos-react";
import {
  type CSSProperties,
  type ReactNode,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { errorText, getJson } from "./api.ts";
import { EditorNote, useEditor } from "./editor.tsx";
import {
  exceptionsOf,
  isTraceback,
  parseTraceback,
  projectPath,
  type TracebackItem,
} from "./traceback.ts";

const LEVELS = ["debug", "info", "warning", "error", "critical"];
const MAX_RECORDS = 5000;
const BOTTOM_SLACK = 24;

interface LogRecord {
  timestamp: string;
  level: string;
  logger: string;
  event: string;
  extra?: Record<string, unknown>;
  raw?: string;
}

const shortTime = (timestamp: string) => /T?(\d\d:\d\d:\d\d)/.exec(timestamp)?.[1] ?? timestamp;
const fieldText = (value: unknown) => typeof value === "string" ? value : JSON.stringify(value);

type OpenFile = ((file: string, line: number) => void) | null;

function FileLink({ file, line, open }: { file: string; line: number; open: OpenFile }) {
  const shown = `${projectPath(file)}:${line}`;
  return open
    ? (
      <button
        type="button"
        className="tb-file"
        title={`${file}:${line} (open in the editor)`}
        data-bp-log-open
        onClick={() => open(projectPath(file), line)}
      >
        {shown}
      </button>
    )
    : <span className="tb-file" title={`${file}:${line}`}>{shown}</span>;
}

function frameRows(frame: Extract<TracebackItem, { kind: "frame" }>, open: OpenFile, key: string) {
  return [
    <div key={key} className={`tb-row tb-frame${frame.library ? " lib" : " own"}`}>
      <span className="tb-gutter">{frame.gutter}</span>
      {frame.indent}
      {frame.library
        ? <span className="tb-path" title={frame.file}>{frame.file}:{frame.line}</span>
        : <FileLink file={frame.file} line={frame.line} open={open} />}
      {frame.func && <span className="tb-func">in {frame.func}</span>}
    </div>,
    ...frame.code.map(({ text, marker }, index) => (
      <div
        key={`${key}.${index}`}
        className={`tb-row ${marker ? "tb-marker" : "tb-code"}${frame.library ? "" : " own"}`}
      >
        <span className="tb-gutter">{frame.gutter}</span>
        {text}
      </div>
    )),
  ];
}

function Traceback({ text, open }: { text: string; open: OpenFile }) {
  const items = parseTraceback(text);
  const rows: ReactNode[] = [];
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (item.kind === "frame" && item.library) {
      let end = i;
      while (items[end + 1]?.kind === "frame" && (items[end + 1] as typeof item).library) {
        end += 1;
      }
      const run = items.slice(i, end + 1) as (typeof item)[];
      rows.push(
        <details key={i} className="tb-lib">
          <summary className="tb-row">
            <span className="tb-gutter">{item.gutter}</span>
            {item.indent}
            {run.length} library frame{run.length === 1 ? "" : "s"}
          </summary>
          {run.flatMap((frame, index) => frameRows(frame, open, `${i}.${index}`))}
        </details>,
      );
      i = end;
    } else if (item.kind === "frame") {
      rows.push(...frameRows(item, open, String(i)));
    } else if (item.kind === "exception") {
      rows.push(
        ...item.lines.map((line, index) => (
          <div key={`${i}.${index}`} className="tb-row tb-exc">
            <span className="tb-gutter">{item.gutter}</span>
            {line}
          </div>
        )),
      );
    } else {
      rows.push(
        <div key={i} className={`tb-row tb-${item.kind}`}>
          <span className="tb-gutter">{item.gutter}</span>
          {item.text}
        </div>,
      );
    }
  }
  const raised = exceptionsOf(items);
  return (
    <div className="tb">
      <div className="tb-lines">{rows}</div>
      {raised.length > 0 && (
        <div className="tb-raised" data-bp-log-raised>
          {raised.map(({ depth, text }, index) => {
            const leaf = !(raised[index + 1]?.depth > depth) && !/^\w*ExceptionGroup:/.test(text);
            return (
              <div
                key={index}
                className={`tb-raised-one${leaf ? " leaf" : ""}`}
                style={{ "--depth": String(depth) } as CSSProperties}
              >
                {text}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Record({ record, open }: { record: LogRecord; open: OpenFile }) {
  const extra = { ...(record.extra ?? {}) };
  const traceback = Array.isArray(extra.traceback_lines)
    ? extra.traceback_lines.join("")
    : isTraceback(extra.exception)
    ? extra.exception
    : null;
  if (traceback) {
    for (const key of ["traceback_lines", "exception", "exception_type", "exception_message"]) {
      delete extra[key];
    }
  }
  const where = [
    extra.lineno !== undefined && `line ${extra.lineno}`,
    extra.func_name && `in ${extra.func_name}`,
  ].filter(Boolean).join(" ");
  delete extra.lineno;
  delete extra.func_name;
  const multiline = (value: unknown) => typeof value === "string" && value.includes("\n");
  const entries = Object.entries(extra);
  const level = record.level.toLowerCase();
  return (
    <div className={`log-rec lv-${level}`} title={traceback ? undefined : record.raw}>
      <div className="log-line">
        <span className="ts">{shortTime(record.timestamp)}</span>
        <span className={`lv ${level}`}>{record.level}</span>
        <span className="ev">
          <span className="lg" title={where ? `${record.logger} ${where}` : record.logger}>
            {record.logger}
          </span>
          {record.event}
          {entries.filter(([, value]) => !multiline(value)).map(([key, value]) => (
            <span key={key} className="chip">
              <span className="k">{key}</span>
              {fieldText(value)}
            </span>
          ))}
        </span>
      </div>
      {entries.filter(([, value]) => multiline(value)).map(([key, value]) => (
        <div key={key} className="log-field">
          <span className="k">{key}</span>
          <pre>{String(value).replace(/\n$/, "")}</pre>
        </div>
      ))}
      {traceback && <Traceback text={traceback} open={open} />}
    </div>
  );
}

export function LogsModal(
  { runId, title, onClose }: { runId: string; title: string; onClose: () => void },
) {
  const [level, setLevel] = useState("info");
  const [query, setQuery] = useState("");
  const [follow, setFollow] = useState(true);
  const [records, setRecords] = useState<LogRecord[]>([]);
  const [hint, setHint] = useState<string | null>("loading…");
  const editor = useEditor();
  const lines = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let offset = 0;
    let timer: ReturnType<typeof setTimeout>;
    let live = true;
    setRecords([]);
    setHint("loading…");
    const poll = async () => {
      const params = new URLSearchParams({ after: String(offset), level });
      if (query) {
        params.set("q", query);
      }
      try {
        const page = await getJson<{ offset: number; records: LogRecord[] }>(
          `runs/${encodeURIComponent(runId)}/log?${params}`,
        );
        if (!live) {
          return;
        }
        offset = page.offset;
        const fresh = page.records;
        setRecords((old) => fresh.length ? [...old, ...fresh].slice(-MAX_RECORDS) : old);
        setHint((old) => fresh.length || old === null ? null : "No log lines yet.");
      } catch (caught) {
        if (live) {
          setHint(errorText(caught));
        }
      }
      if (live) {
        timer = setTimeout(poll, 1500);
      }
    };
    timer = setTimeout(poll, query ? 250 : 0);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [runId, level, query]);
  useLayoutEffect(() => {
    if (follow && lines.current) {
      lines.current.scrollTop = lines.current.scrollHeight;
    }
  }, [records, follow]);
  useEffect(() => {
    // opening a library frame means reading back up: stop following (toggle doesn't bubble, so capture it)
    const box = lines.current!;
    const opened = (event: Event) => (event.target as HTMLDetailsElement).open && setFollow(false);
    box.addEventListener("toggle", opened, true);
    return () => box.removeEventListener("toggle", opened, true);
  }, []);
  const atBottom = () => {
    const box = lines.current!;
    return box.scrollHeight - box.scrollTop - box.clientHeight <= BOTTOM_SLACK;
  };
  return (
    <Modal
      open
      onClose={onClose}
      size="lg"
      className="logs"
      title={
        <>
          Logs · {title} <span className="cfg-sub">{runId}</span>
        </>
      }
      data-bp-logs-modal
    >
      <div className="log-tools">
        <Select
          size="sm"
          aria-label="lowest level"
          value={level}
          onChange={setLevel}
          options={LEVELS.map((name) => [name, `${name} and up`] as const)}
        />
        <SearchInput
          placeholder="search the log…"
          aria-label="search the log"
          value={query}
          onChange={setQuery}
        />
        <Checkbox label="follow" value={follow} onChange={setFollow} />
      </div>
      <EditorNote status={editor.status} />
      <div
        ref={lines}
        className="log-body"
        onScroll={() => atBottom() !== follow && setFollow(atBottom())}
      >
        {hint && <div className="hint">{hint}</div>}
        <div className="log-content">
          {records.map((record, index) => (
            <Record
              key={index}
              record={record}
              open={editor.open}
            />
          ))}
        </div>
      </div>
    </Modal>
  );
}

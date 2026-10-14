// A run's structured log (GET /runs/{id}/log), followed live; the server filters by level and search.
import { EditorNote, type LogRecord, Modal, StructuredLog, useDesktopEditor } from "dimos-react";
import { useEffect, useState } from "react";
import { errorText, getJson } from "./api.ts";
import { projectPath } from "./model.ts";

const MAX_RECORDS = 5000;

export function LogsModal(
  { runId, title, onClose }: { runId: string; title: string; onClose: () => void },
) {
  const [level, setLevel] = useState("info");
  const [query, setQuery] = useState("");
  const [follow, setFollow] = useState(true);
  const [records, setRecords] = useState<LogRecord[]>([]);
  const [note, setNote] = useState<string | null>("loading…");
  const editor = useDesktopEditor();
  useEffect(() => {
    let offset = 0;
    let timer: ReturnType<typeof setTimeout>;
    let live = true;
    setRecords([]);
    setNote("loading…");
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
        setNote((old) => fresh.length || old === null ? null : "No log lines yet.");
      } catch (caught) {
        if (live) {
          setNote(errorText(caught));
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
  return (
    <Modal
      open
      onClose={onClose}
      size="xl"
      title={
        <>
          Logs · {title} <span className="sub">{runId}</span>
        </>
      }
      data-bp-logs-modal
    >
      <StructuredLog
        className="fill"
        records={records}
        note={note}
        level={level}
        onLevelChange={setLevel}
        query={query}
        onQueryChange={setQuery}
        follow={follow}
        onFollowChange={setFollow}
        banner={<EditorNote status={editor.status} />}
        onOpenFile={editor.open ?? undefined}
        formatPath={projectPath}
      />
    </Modal>
  );
}

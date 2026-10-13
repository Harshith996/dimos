// A module's or the blueprint's source, over the graph, scrolled to its line.
import { Button, CodeBlock } from "dimos-react";
import { useEffect, useRef, useState } from "react";
import { errorText, getJson } from "./api.ts";
import { EditorNote, useEditor } from "./editor.tsx";
import type { Where } from "./side.tsx";

export function CodePane({ where, onClose }: { where: Where; onClose: () => void }) {
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const editor = useEditor();
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let current = true;
    setText(null);
    setError(null);
    getJson<{ text: string }>(`source?file=${encodeURIComponent(where.file)}`).then(
      (answer) => current && setText(answer.text),
      (caught) => current && setError(errorText(caught)),
    );
    return () => {
      current = false;
    };
  }, [where.file]);
  useEffect(() => {
    const lines = box.current?.querySelectorAll<HTMLElement>(".dr-code__line");
    const target = lines?.[where.line - 1];
    lines?.forEach((line) => line.classList.toggle("at", line === target));
    const pre = box.current?.querySelector<HTMLElement>(".dr-code__pre");
    if (target && pre) {
      pre.scrollTop = target.offsetTop - pre.offsetTop - 8;
    }
  }, [text, where.line]);
  return (
    <div id="code" ref={box}>
      <div className="code-head">
        <span className="file" title={where.file}>{where.file}</span>
        <span className="at">line {where.line}</span>
        <span className="spacer" />
        {editor.open && (
          <Button
            size="sm"
            onClick={() => editor.open?.(where.file, where.line)}
            data-bp-open-editor
          >
            Open in editor
          </Button>
        )}
        <Button size="sm" onClick={onClose}>Hide code</Button>
      </div>
      <EditorNote status={editor.status} />
      {error && <div className="note">{error}</div>}
      {text === null && !error && <div className="note">loading…</div>}
      {text !== null && (
        <CodeBlock code={text} language="py" lineNumbers copy={false} title={null} />
      )}
    </div>
  );
}

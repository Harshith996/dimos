// Desktop opens a file in the user's editor: the view asks with dimos:open-in-editor and hears back with
// dimos:open-in-editor-result. Only framed by Desktop; off Desktop there is no editor to ask.
import { useEffect, useState } from "react";
import { FRAMED, toDesktop } from "./api.ts";

export interface EditorStatus {
  ok: boolean;
  text: string;
}

/** `open` asks Desktop to open a file (null when not framed); `status` is its last answer. */
export function useEditor(): {
  open: ((file: string, line: number) => void) | null;
  status: EditorStatus | null;
} {
  const [status, setStatus] = useState<EditorStatus | null>(null);
  useEffect(() => {
    const listen = (event: MessageEvent) => {
      if (event.origin !== location.origin || event.source !== parent) {
        return;
      }
      if (event.data?.type === "dimos:open-in-editor-result") {
        setStatus({ ok: !!event.data.ok, text: String(event.data.text ?? "") });
      }
    };
    addEventListener("message", listen);
    return () => removeEventListener("message", listen);
  }, []);
  const open = (file: string, line: number) => {
    setStatus(null);
    toDesktop({ type: "dimos:open-in-editor", file, line });
  };
  return { open: FRAMED ? open : null, status };
}

export function EditorNote({ status }: { status: EditorStatus | null }) {
  return status
    ? <div className={`opened${status.ok ? "" : " failed"}`} title={status.text}>{status.text}</div>
    : null;
}

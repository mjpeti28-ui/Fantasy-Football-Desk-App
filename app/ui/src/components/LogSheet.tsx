import { useEffect, useRef, useState } from "react";
import type { JDraft, JStatus } from "../types";
import { pct } from "../format";

const STATUS_OPTS: Record<string, { id: JStatus; label: string }[]> = {
  trade: [{ id: "offered", label: "Offering it" }, { id: "done", label: "It went through" }, { id: "skipped", label: "Passing" }],
  other: [{ id: "done", label: "Doing it" }, { id: "skipped", label: "Not doing it" }],
};

/** One sheet for every "Log to journal": a title, your reason, and how sure you are, next to what
 *  the model said. Field-notes look, same as the Journal screen. */
export function LogSheet({ draft, onCancel, onSave }: { draft: JDraft; onCancel: () => void; onSave: (d: JDraft) => Promise<void> }) {
  const [title, setTitle] = useState(draft.title);
  const [why, setWhy] = useState(draft.why ?? "");
  const [conf, setConf] = useState<number>(draft.confidence ?? 0.65);
  const [noConf, setNoConf] = useState(draft.kind === "note");
  const opts = STATUS_OPTS[draft.kind === "trade" ? "trade" : "other"];
  const [status, setStatus] = useState<JStatus>(draft.status ?? opts[0].id);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const area = useRef<HTMLTextAreaElement>(null);
  useEffect(() => { area.current?.focus(); }, []);
  const save = async () => {
    if (!title.trim()) return;
    setBusy(true); setErr(null);
    try { await onSave({ ...draft, title, why, confidence: noConf ? null : conf, status }); }
    catch (e) { setErr(String(e)); setBusy(false); }
  };
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
      if (e.key === "Enter" && e.metaKey) { e.preventDefault(); void save(); }
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  });
  return (
    <div className="logsheet-wrap" role="presentation" onMouseDown={(e) => { if (e.target === e.currentTarget) onCancel(); }}>
      <div className="logsheet" role="dialog" aria-modal="true" aria-label="Log to journal">
        <span className="kind">{draft.kind === "note" ? "Note" : `${draft.kind[0].toUpperCase()}${draft.kind.slice(1)} decision`}{draft.week ? ` · week ${draft.week}` : ""}</span>
        <input className="ttl" value={title} onChange={(e) => setTitle(e.target.value)} aria-label="Title" />
        {draft.context && <p className="ctx">{draft.context}</p>}
        <textarea ref={area} value={why} onChange={(e) => setWhy(e.target.value)} placeholder="Why? What do you see that the numbers might not?" rows={4} aria-label="Why" />
        {draft.kind !== "note" && (
          <div className="row">
            <label className="sure-ctl">
              <span>How sure are you?</span>
              <input type="range" min={50} max={95} step={5} value={Math.round(conf * 100)} disabled={noConf} onChange={(e) => setConf(Number(e.target.value) / 100)} />
              <b className="num">{noConf ? "–" : pct(conf)}</b>
            </label>
            {draft.model_p != null && <span className="mp">model <b className="num">{pct(draft.model_p)}</b></span>}
          </div>
        )}
        {draft.kind !== "note" && (
          <div className="row">
            <div className="seg" role="group" aria-label="Status">
              {opts.map((o) => <button key={o.id} type="button" aria-pressed={status === o.id} onClick={() => setStatus(o.id)}>{o.label}</button>)}
            </div>
            <label className="skip"><input type="checkbox" checked={noConf} onChange={(e) => setNoConf(e.target.checked)} /> No confidence</label>
          </div>
        )}
        {err && <p className="err">{err}</p>}
        <div className="acts">
          <span className="dim">Only decisions you made count toward your calibration.</span>
          <button className="tbtn" type="button" onClick={onCancel}>Cancel</button>
          <button className="tbtn pri" type="button" onClick={save} disabled={busy || !title.trim()}>Log it <kbd>⌘↵</kbd></button>
        </div>
      </div>
    </div>
  );
}

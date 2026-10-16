import { useEffect, useMemo, useRef, useState } from "react";
import type { Pos } from "../types";
import { POS_VAR } from "./bits";
import "./palette.css";

/* ⌘K — a launcher. Keyboard first: type, arrow, return. Players open their file (⌘↵ queues them
   for Compare), teams jump to the League or a trade, and the rest are the app's own commands. */

/** owner: a team name, null = free agent, undefined = not known yet (rosters still loading) */
export interface PalPlayer { id: number; name: string; pos: Pos; team: string | null; owner?: string | null; mine: boolean; note?: string }
export interface PalAction { id: string; label: string; hint?: string; group: "Go to" | "Teams" | "Actions"; run: () => void }

function score(q: string, name: string): number {
  const n = name.toLowerCase(), parts = n.split(/[\s.'-]+/);
  if (n.startsWith(q)) return 100 - n.length / 100;
  if (parts.some((p) => p.startsWith(q))) return 80 - n.length / 100;
  if (n.includes(q)) return 50;
  // loose fallback, 3+ letters: initials ("bsb" finds Blue Streak Bombers) or the letters in order
  // starting at a word start. Anywhere-in-the-name matching made "lamb" find Blue Streak Bombers.
  if (q.length < 3) return -1;
  if (parts.map((w) => w[0] ?? "").join("").includes(q)) return 30;
  for (let k = 0; k < n.length; k++) {
    if (n[k] !== q[0] || (k > 0 && !/[\s.'-]/.test(n[k - 1]))) continue;
    let i = 0;
    for (let j = k; j < n.length && i < q.length; j++) if (n[j] === q[i]) i++;
    if (i === q.length) return 20;
  }
  return -1;
}

export function Palette({ players, actions, onClose, onOpen, onCompare }: {
  players: PalPlayer[]; actions: PalAction[]; onClose: () => void; onOpen: (id: number) => void; onCompare: (id: number) => void;
}) {
  const [q, setQ] = useState("");
  const [i, setI] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLUListElement>(null);
  useEffect(() => { input.current?.focus(); }, []);
  const rows = useMemo(() => {
    const t = q.trim().toLowerCase();
    type R = { kind: "player"; p: PalPlayer; s: number } | { kind: "action"; a: PalAction; s: number };
    const out: R[] = [];
    if (!t) {
      for (const a of actions.filter((a) => a.group !== "Teams")) out.push({ kind: "action", a, s: 0 });
      return out;
    }
    // filter on the raw match first: the "yours" bonus only orders matches, it must not let a non-match through
    const ps = players.map((p) => ({ kind: "player" as const, p, m: score(t, p.name) })).filter((r) => r.m >= 0)
      .map((r) => ({ kind: r.kind, p: r.p, s: r.m + (r.p.mine ? 5 : 0) })).sort((a, b) => b.s - a.s).slice(0, 8);
    const as = actions.map((a) => ({ kind: "action" as const, a, s: score(t, a.label) })).filter((r) => r.s >= 0).sort((a, b) => b.s - a.s).slice(0, 6);
    return [...ps, ...as];
  }, [q, players, actions]);
  useEffect(() => { setI(0); }, [q]);
  useEffect(() => { list.current?.querySelector<HTMLElement>(`[data-i="${i}"]`)?.scrollIntoView({ block: "nearest" }); }, [i]);
  const go = (k: number, meta: boolean) => {
    const r = rows[k];
    if (!r) return;
    if (r.kind === "player") { if (meta && r.p.pos !== "K" && r.p.pos !== "DST") onCompare(r.p.id); else onOpen(r.p.id); }
    else r.a.run();
    onClose();
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") { e.preventDefault(); onClose(); }
    else if (e.key === "ArrowDown") { e.preventDefault(); setI((x) => Math.min(rows.length - 1, x + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setI((x) => Math.max(0, x - 1)); }
    else if (e.key === "Enter") { e.preventDefault(); go(i, e.metaKey); }
  };
  let lastGroup = "";
  return (
    <div className="pal-scrim" onMouseDown={onClose}>
      <div className="pal" role="dialog" aria-modal="true" aria-label="Command palette" onMouseDown={(e) => e.stopPropagation()} onKeyDown={onKey}>
        <input ref={input} value={q} onChange={(e) => setQ(e.target.value)} placeholder="Find a player, team or command…" aria-label="Search"
          role="combobox" aria-expanded="true" aria-controls="pal-list" aria-activedescendant={rows.length ? `pal-${i}` : undefined} />
        <ul id="pal-list" ref={list} role="listbox">
          {rows.map((r, k) => {
            const g = r.kind === "player" ? "Players" : r.a.group;
            const head = g !== lastGroup ? <li className="g" role="presentation">{g}</li> : null;
            lastGroup = g;
            return [head, (
              <li key={r.kind === "player" ? `p${r.p.id}` : r.a.id} id={`pal-${k}`} data-i={k} role="option" aria-selected={k === i}
                className={k === i ? "on" : ""} onMouseMove={() => setI(k)} onClick={(e) => go(k, e.metaKey)}>
                {r.kind === "player" ? <>
                  <i className="pchip" style={{ background: POS_VAR[r.p.pos], color: r.p.pos === "WR" ? "var(--wr-ink)" : "var(--pos-ink)" }}>{r.p.pos}</i>
                  <b>{r.p.name}</b>
                  <span className="meta">{r.p.team ?? ""}{r.p.mine || r.p.owner !== undefined ? <>{" · "}<span className={r.p.mine ? "mine" : r.p.owner ? "" : "fa"}>{r.p.mine ? "yours" : r.p.owner ?? "free agent"}</span></> : null}{r.p.note ? ` · ${r.p.note}` : ""}</span>
                  <kbd>{k === i ? (r.p.pos === "K" || r.p.pos === "DST" ? "↵ file" : "↵ file · ⌘↵ compare") : ""}</kbd>
                </> : <>
                  <b>{r.a.label}</b>{r.a.hint && <span className="meta">{r.a.hint}</span>}<kbd>{k === i ? "↵" : ""}</kbd>
                </>}
              </li>
            )];
          })}
          {!rows.length && <li className="none">Nothing matches “{q}”.</li>}
        </ul>
        <p className="pal-foot"><kbd>↑↓</kbd> move <kbd>↵</kbd> open <kbd>⌘↵</kbd> add to compare <kbd>esc</kbd> close</p>
      </div>
    </div>
  );
}

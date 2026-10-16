import { useEffect, useRef } from "react";
import type { ChangeItem, Changes } from "../data";
import { POS_VAR } from "./bits";
import "./changes.css";

/* "What's new": a short ticker of what moved between the pull you last looked at and the newest
   one, grouped by how much it matters to you. Green/red = good/bad for you (a tag on their
   player going bad is good news), never just up/down. */

const when = (t?: number | null) => (t ? new Date(t * 1000).toLocaleString(undefined, { weekday: "short", hour: "numeric", minute: "2-digit" }) : "");

export function ChangesPanel({ data, teamName, onOpen, onClose }: {
  data: Changes; teamName: (id: number | null) => string; onOpen: (id: number) => void; onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    const m = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node) && !(e.target as HTMLElement).closest?.(".chg-btn")) onClose(); };
    window.addEventListener("keydown", k); window.addEventListener("mousedown", m);
    return () => { window.removeEventListener("keydown", k); window.removeEventListener("mousedown", m); };
  }, [onClose]);
  const groups: { key: ChangeItem["kind"]; title: string }[] = [
    { key: "mine", title: "Your team" }, { key: "opp", title: data.opp_name ? `This week's opponent · ${data.opp_name}` : "This week's opponent" }, { key: "move", title: "League moves" },
  ];
  const row = (it: ChangeItem, i: number) => (
    <li key={`${it.kind}${it.id}${i}`}>
      <i className={`tone ${it.tone}`} aria-label={it.tone === "up" ? "good for you" : it.tone === "down" ? "bad for you" : "neutral"} />
      {it.pos && <i className="pchip" style={{ background: POS_VAR[it.pos as keyof typeof POS_VAR] ?? "var(--field)", color: it.pos === "WR" ? "var(--wr-ink)" : "var(--pos-ink)" }}>{it.pos}</i>}
      <button type="button" className="nm" onClick={() => onOpen(it.id)} title="Open player file">{it.name}</button>
      <span className="tx">{it.kind === "move" && it.text.startsWith(teamName(it.team)) ? it.text.slice(teamName(it.team).length).trim() : it.text}</span>
    </li>
  );
  return (
    <div className="chg-panel" ref={ref} role="dialog" aria-label="What's new">
      <header><b>What's new</b><span>{data.at_from ? `since ${data.from_source === "weekly run" ? "the weekly run" : "your last look"}, ${when(data.at_from)}` : ""}{data.at_to ? ` → pull of ${when(data.at_to)}` : ""}</span></header>
      {!data.items.length && <p className="empty">{data.note ?? "Nothing changed that touches your team, your opponent or the league's rosters."}</p>}
      {groups.map((g) => {
        const xs = data.items.filter((x) => x.kind === g.key);
        if (!xs.length) return null;
        return (
          <section key={g.key}>
            <h6>{g.title}</h6>
            <ul>{g.key === "move"
              ? [...new Set(xs.map((x) => x.team))].flatMap((t) => [<li key={`t${t}`} className="team">{teamName(t)}</li>, ...xs.filter((x) => x.team === t).map(row)])
              : xs.map(row)}</ul>
          </section>
        );
      })}
      <footer>Injury tags and projections are ESPN's; roster moves are between your last look and the newest pull. Desk refreshes on its own every few hours, and every 25 minutes in the hours before your games kick off.</footer>
    </div>
  );
}

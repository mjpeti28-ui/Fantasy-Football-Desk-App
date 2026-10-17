import { useState } from "react";
import "./journal.css";
import type { JDraft, JEntry, JKind, JStatus, Journal as JournalT } from "../../types";
import { f1, pct, signed } from "../../format";
import { useFile } from "../../components/FileContext";
import type { Prompt } from "./prompts";

type Filter = JKind | "all";
const KIND_LABEL: Record<JKind, string> = { lineup: "Lineup", waiver: "Waiver", trade: "Trade", note: "Note" };
const STATUS_LABEL: Record<JStatus, string> = { done: "Did it", offered: "Offered", skipped: "Didn't" };

const scored = (e: JEntry) => e.outcome.state === "final" && e.outcome.right != null && e.status === "done";

function Calibration({ entries }: { entries: JEntry[] }) {
  const done = entries.filter(scored);
  const W = 300, H = 190, L = 34, B = 26, T = 10, R = 10;
  const x = (c: number) => L + ((c - 0.5) / 0.5) * (W - L - R);
  const y = (h: number) => T + (1 - h) * (H - T - B);
  const bucket = (key: "confidence" | "model_p") => [[0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 1.01]].map(([lo, hi]) => {
    const b = done.filter((e) => e[key] != null && e[key]! >= lo && e[key]! < hi);
    if (!b.length) return null;
    return { c: b.reduce((s, e) => s + e[key]!, 0) / b.length, hit: b.filter((e) => e.outcome.right).length / b.length, n: b.length };
  }).filter((v): v is { c: number; hit: number; n: number } => v != null);
  const mine = bucket("confidence"), model = bucket("model_p");
  return (
    <svg className="calib" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Calibration: how often you were right at each confidence level">
      {[0, 0.25, 0.5, 0.75, 1].map((h) => <g key={h}><line x1={L} x2={W - R} y1={y(h)} y2={y(h)} className="grid" /><text x={L - 6} y={y(h) + 3.5} textAnchor="end">{Math.round(h * 100)}</text></g>)}
      {[0.5, 0.6, 0.7, 0.8, 0.9, 1].map((c) => <text key={c} x={x(c)} y={H - 8} textAnchor="middle">{Math.round(c * 100)}</text>)}
      <line x1={x(0.5)} y1={y(0.5)} x2={x(1)} y2={y(1)} className="diag" />
      <text x={x(1)} y={y(0.9) + 14} className="diag-l" textAnchor="end">perfectly calibrated</text>
      {model.map((m, i) => <rect key={`m${i}`} x={x(m.c) - 4.5} y={y(m.hit) - 4.5} width={9} height={9} transform={`rotate(45 ${x(m.c)} ${y(m.hit)})`} className="model" />)}
      {mine.map((m, i) => <circle key={`y${i}`} cx={x(m.c)} cy={y(m.hit)} r={3.5 + 2 * Math.sqrt(m.n)} className="you"><title>{`You said ~${pct(m.c)}: right ${Math.round(m.hit * m.n)} of ${m.n}`}</title></circle>)}
    </svg>
  );
}

function verdictText(j: JournalT): string {
  const done = j.entries.filter(scored);
  if (!done.length) return "Nothing scored yet. Lineup calls are scored when their week's games finish; waiver and trade calls keep a running total until week 17.";
  const hits = done.filter((e) => e.outcome.right).length;
  const conf = done.filter((e) => e.confidence != null);
  const avg = conf.length ? conf.reduce((s, e) => s + e.confidence!, 0) / conf.length : null;
  const mp = done.filter((e) => e.model_p != null);
  const mavg = mp.length ? mp.reduce((s, e) => s + e.model_p!, 0) / mp.length : null;
  let t = `Right on ${hits} of ${done.length} scored calls (${pct(hits / done.length)}).`;
  if (avg != null) t += ` You said ${pct(avg)} on average`;
  if (mavg != null) t += `${avg != null ? "; the model said" : " The model said"} ${pct(mavg)}`;
  if (avg != null || mavg != null) t += ".";
  t += done.length < 20 ? ` ${done.length} calls is too few to call anyone over- or underconfident; about 20 starts to mean something.`
    : avg != null && avg - hits / done.length > 0.08 ? " You've been overconfident." : avg != null && hits / done.length - avg > 0.08 ? " You've been underconfident." : " That's well calibrated.";
  return t;
}

function Margin({ e }: { e: JEntry }) {
  const o = e.outcome, m = e.model as Record<string, number | string | undefined>;
  const lines: JSX.Element[] = [];
  if (e.model_p != null) lines.push(<div key="p"><span>model</span><b className="num">{pct(e.model_p)}</b></div>);
  if (typeof m.gain === "number") lines.push(<div key="g"><span>model gain</span><b className="num">{signed(m.gain)}</b></div>);
  if (typeof m.gain_cal === "number") lines.push(<div key="gc"><span>model gain</span><b className="num">{signed(m.gain_cal)}</b></div>);
  if (typeof m.espn_partner === "number") lines.push(<div key="ep"><span>their ESPN view</span><b className="num">{signed(m.espn_partner)}</b></div>);
  let out: JSX.Element | null = null;
  const would = e.status === "skipped" ? "would have been " : "";
  if (o.state === "pending") out = <div className="oc pend">Scored after week {e.week}</div>;
  else if (o.state === "final" && e.kind === "lineup") out = (
    <div className={`oc ${o.right ? "ok" : "bad"}`}><b>{would}{o.right ? "Right" : "Wrong"}</b><span className="num">{f1(o.start_pts)} started vs {f1(o.bench_pts)} benched</span></div>);
  else if (o.state === "running" || o.state === "final") out = (
    <div className={`oc ${o.state === "running" ? "run" : o.right ? "ok" : "bad"}`}>
      <b>{o.state === "running" ? `Weeks ${o.weeks?.[0]}–${o.weeks?.[1]}, so far` : `${would}${o.right ? "Right" : "Wrong"}`}</b>
      <span className="num">{e.kind === "trade" ? "got" : "added"} {f1(o.in_pts)} · {e.kind === "trade" ? "gave" : "dropped"} {f1(o.out_pts)}</span>
      <span className="num diff">{signed((o.in_pts ?? 0) - (o.out_pts ?? 0))}</span>
    </div>);
  if (!lines.length && !out) return <aside className="margin" />;
  return <aside className="margin">{lines}{out}</aside>;
}

function People({ e, names }: { e: JEntry; names: Record<string, string> }) {
  const { open } = useFile();
  const p = e.payload as Record<string, unknown>;
  const ids = [p.start_id, p.bench_id, ...((p.add_ids as number[]) ?? []), ...((p.drop_ids as number[]) ?? []), ...((p.get_ids as number[]) ?? []),
    ...((p.give_ids as number[]) ?? []), ...((p.player_ids as number[]) ?? [])].filter((x): x is number => typeof x === "number");
  const uniq = [...new Set(ids)].filter((id) => names[String(id)]);
  if (!uniq.length) return null;
  return <span className="people">{uniq.map((id) => <button key={id} type="button" className="lnk" onClick={() => open(id, uniq)}>{names[String(id)]} file</button>)}</span>;
}

function Entry({ e, onUpdate, onDelete, names }: { e: JEntry; onUpdate: (id: string, f: Partial<JEntry>) => Promise<void>; onDelete: (id: string) => Promise<void>; names: Record<string, string> }) {
  const [edit, setEdit] = useState(false);
  const [why, setWhy] = useState(e.why);
  const [conf, setConf] = useState(e.confidence);
  const [del, setDel] = useState(false);
  const opts: JStatus[] = e.kind === "trade" ? ["offered", "done", "skipped"] : ["done", "skipped"];
  return (
    <article className={`entry k-${e.kind} ${e.status === "skipped" ? "skipped" : ""}`}>
      <div className="ebody">
        <span className="kind">{KIND_LABEL[e.kind]}{e.kind !== "note" ? ` · ${STATUS_LABEL[e.status]}` : ""}</span>
        <h4>{e.title}</h4>
        {edit ? (
          <div className="edit">
            <textarea value={why} onChange={(x) => setWhy(x.target.value)} rows={3} aria-label="Why" />
            {e.kind !== "note" && <label className="sure-ctl"><span>How sure</span><input type="range" min={50} max={95} step={5} value={Math.round((conf ?? 0.65) * 100)} onChange={(x) => setConf(Number(x.target.value) / 100)} /><b className="num">{pct(conf)}</b></label>}
            <div className="acts">
              <button className="tbtn pri" type="button" onClick={async () => { await onUpdate(e.id, { why, confidence: conf }); setEdit(false); }}>Save</button>
              <button className="tbtn" type="button" onClick={() => { setWhy(e.why); setConf(e.confidence); setEdit(false); }}>Cancel</button>
            </div>
          </div>
        ) : e.why ? <p className="why">{e.why}</p> : <p className="why none">No reason written.</p>}
        <div className="meta">
          {e.confidence != null && <span className="sure">you said <b className="num">{pct(e.confidence)}</b></span>}
          {e.kind !== "note" && (
            <span className="status-pick" role="group" aria-label="Did it happen">
              {opts.map((s) => <button key={s} type="button" aria-pressed={e.status === s} onClick={() => void onUpdate(e.id, { status: s })}>{STATUS_LABEL[s]}</button>)}
            </span>
          )}
          <People e={e} names={names} />
          {!edit && <button type="button" className="lnk" onClick={() => setEdit(true)}>Edit</button>}
          {del ? <span className="confirm">Delete this entry? <button type="button" className="lnk bad" onClick={() => void onDelete(e.id)}>Delete</button> <button type="button" className="lnk" onClick={() => setDel(false)}>Keep</button></span>
            : <button type="button" className="lnk" onClick={() => setDel(true)}>Delete</button>}
          <span className="when">{new Date(e.created_at * 1000).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" })}</span>
        </div>
      </div>
      <Margin e={e} />
    </article>
  );
}

export function Journal({ journal, err, engineUp, onAdd, onUpdate, onDelete, names = {}, prompts = [], onLog }: {
  journal: JournalT | null; err: string | null; engineUp: boolean; names?: Record<string, string>; prompts?: Prompt[]; onLog?: (d: JDraft) => void;
  onAdd: (d: JDraft) => Promise<void>; onUpdate: (id: string, f: Partial<JEntry>) => Promise<void>; onDelete: (id: string) => Promise<void>;
}) {
  const [filter, setFilter] = useState<Filter>("all");
  const [note, setNote] = useState("");
  if (!engineUp) return <div className="journal"><header className="jr-hd"><h2>Journal</h2><p>The journal lives in the engine. Start it with app/run.sh to read and write entries.</p></header></div>;
  if (err) return <div className="journal"><p className="jr-err">{err}</p></div>;
  if (!journal) return <div className="journal"><header className="jr-hd"><h2>Journal</h2><p>Opening…</p></header></div>;
  const rows = journal.entries.filter((e) => filter === "all" || e.kind === filter);
  const weeks = [...new Set(rows.map((e) => e.week))].sort((a, b) => b - a);
  const s = journal.summary;
  const addNote = async () => {
    const t = note.trim(); if (!t) return;
    const [first, ...rest] = t.split("\n");
    await onAdd({ kind: "note", title: first.slice(0, 120), why: rest.join("\n").trim() || (first.length > 120 ? first : ""), week: journal.week });
    setNote("");
  };
  return (
    <div className="journal">
      <header className="jr-hd">
        <h2>Journal</h2>
        <p>What you decided, why, and how sure you were, next to what the model said. Scored against what happened.</p>
      </header>
      <section className="score">
        <div className="figs">
          <div><b className="num">{s.logged}</b><span>logged</span></div>
          <div><b className="num">{s.resolved}</b><span>scored</span></div>
          <div><b className="num">{s.hit_rate == null ? "–" : s.resolved < 10 ? `${Math.round(s.hit_rate * s.resolved)} of ${s.resolved}` : pct(s.hit_rate)}</b><span>right</span></div>
          <p>{verdictText(journal)}</p>
        </div>
        <figure>
          <Calibration entries={journal.entries} />
          <figcaption><i className="k-you" /> you, by how sure you said you were <i className="k-model" /> the model · across: stated %, up: % right</figcaption>
        </figure>
      </section>
      {prompts.length > 0 && onLog && (
        <section className="prompts" aria-label="Worth logging this week">
          <h3>Worth logging this week <span>the calls where your read could beat the model, or lose to it</span></h3>
          <ol>
            {prompts.map((p) => (
              <li key={p.key}>
                <span className="pk">{p.kind === "lineup" ? "Lineup" : p.kind === "waiver" ? "Waiver" : "Hunch"}</span>
                <span className="pt"><b>{p.title}</b><span>{p.why}</span></span>
                <button className="tbtn" type="button" onClick={() => onLog(p.draft)}>Log it…</button>
              </li>
            ))}
          </ol>
        </section>
      )}
      <section className="compose">
        <textarea value={note} onChange={(e) => setNote(e.target.value)} placeholder="A note for this week: a hunch, a read on a player, something to check later. First line is the title."
          rows={2} onKeyDown={(e) => { if (e.key === "Enter" && e.metaKey) { e.preventDefault(); void addNote(); } }} aria-label="New note" />
        <button className="tbtn" type="button" onClick={addNote} disabled={!note.trim()}>Add note <kbd>⌘↵</kbd></button>
      </section>
      <div className="jr-tools">
        <div className="seg" role="group" aria-label="Kind">
          {(["all", "lineup", "waiver", "trade", "note"] as Filter[]).map((k) => <button key={k} type="button" aria-pressed={filter === k} onClick={() => setFilter(k)}>{k === "all" ? "All" : KIND_LABEL[k]}</button>)}
        </div>
        <span className="dim">Log decisions from My Team, Trades and Waivers with “Log to journal”.</span>
      </div>
      {!rows.length && <p className="jr-empty">{journal.entries.length ? "Nothing of this kind yet." : "No entries yet. The first one is the hardest: log this week's closest lineup call from My Team."}</p>}
      {weeks.map((w) => (
        <section key={w} className="wk">
          <h3>Week {w}{w === journal.week ? <small> · this week</small> : null}</h3>
          {rows.filter((e) => e.week === w).map((e) => <Entry key={e.id} e={e} onUpdate={onUpdate} onDelete={onDelete} names={names} />)}
        </section>
      ))}
    </div>
  );
}

export function journalMarkdown(j: JournalT): string {
  const L = ["## Decision journal (from Desk)", "", verdictText(j), ""];
  for (const e of j.entries) {
    const o = e.outcome;
    const res = o.state === "final" && e.kind === "lineup" ? ` → ${o.right ? "right" : "wrong"} (${f1(o.start_pts)} vs ${f1(o.bench_pts)})`
      : o.state === "running" || o.state === "final" ? ` → in ${f1(o.in_pts)} / out ${f1(o.out_pts)}` : "";
    L.push(`- Wk ${e.week} ${KIND_LABEL[e.kind]}${e.kind !== "note" ? ` (${STATUS_LABEL[e.status]})` : ""}: ${e.title}${e.confidence != null ? `, you ${pct(e.confidence)}` : ""}${e.model_p != null ? `, model ${pct(e.model_p)}` : ""}${res}${e.why ? `. ${e.why}` : ""}`);
  }
  return L.join("\n");
}

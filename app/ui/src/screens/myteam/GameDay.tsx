import { useFile } from "../../components/FileContext";
import { useMemo } from "react";
import type { Game, Player, TeamWeek } from "../../types";
import { clock, f1, lastName, teamShort } from "../../format";
import { isDead } from "../../components/bits";
import type { LiveBoard, LivePlayer } from "../live/Live";
import { STARTERS, type Plan, slotOf } from "./lineup";
import "./gameday.css";

interface Row { id: number; name: string; pos: string; team: string | null; proj: number; game: Game; slot?: string }
const key = (g: Game) => `${g.date} ${g.time}`;

export function GameDay({ tw, plan, onSelect, selected, live }: { tw: TeamWeek; plan: Plan; onSelect: (id: number) => void; selected: number | null; live?: LiveBoard | null }) {
  const { open: openFile } = useFile();
  // started and finished games come from the live feed (same week only), keyed by player
  const lp = useMemo(() => {
    const m = new Map<number, LivePlayer>();
    if (live && live.week === tw.week) for (const p of [...live.me, ...live.opp]) m.set(p.id, p);
    return m;
  }, [live, tw.week]);
  const by = useMemo(() => new Map(tw.players.map((p) => [p.id, p])), [tw]);
  const mine: Row[] = STARTERS.filter((s) => plan[s] != null).map((s) => by.get(plan[s]!)!).filter((p) => p.game && !p.game.bye)
    .map((p) => ({ id: p.id, name: p.name, pos: p.pos, team: p.team, proj: p.espn_week ?? 0, game: p.game!, slot: slotOf(plan, p.id) }));
  const theirs: Row[] = (tw.matchup?.opp_starters ?? []).filter((p) => p.game && !p.game.bye)
    .map((p) => ({ id: p.id, name: p.name, pos: p.pos, team: p.team, proj: p.espn_week ?? 0, game: p.game!, slot: p.slot }));
  const reserves: Player[] = tw.players.filter((p) => !slotOf(plan, p.id) && !isDead(p) && p.game && !p.game.bye);
  const wins = [...new Set([...mine, ...theirs, ...reserves.map((p) => ({ game: p.game! }))].map((r) => key(r.game)))].sort();
  let cm = 0, ct = 0;
  const cum = wins.map((w) => {
    cm += mine.filter((r) => key(r.game) === w).reduce((a, r) => a + r.proj, 0);
    ct += theirs.filter((r) => key(r.game) === w).reduce((a, r) => a + r.proj, 0);
    return [cm, ct] as const;
  });
  const total = cm, mx = Math.max(cm, ct, 1);
  const W = 1000, H = 120, L = 36, R = 96, T = 22, B = 20;
  const x = (i: number) => L + ((W - L - R) * (i + 0.5)) / wins.length, y = (v: number) => T + (H - T - B) * (1 - v / mx);
  const path = (k: 0 | 1) => cum.map((c, i) => `${i ? "L" : "M"}${x(i)},${y(c[k])}`).join(" ");
  const half = cum.findIndex((c) => c[0] >= total / 2);
  // games where both lineups have starters: the same game swings both sides
  const gid = (r: Row) => `${r.game.date}|${[r.team, r.game.opp].sort().join("-")}`;
  const gname = (r: Row) => (r.game.home ? `${r.game.opp} @ ${r.team}` : `${r.team} @ ${r.game.opp}`);
  const shared = [...new Set(mine.map(gid))].filter((g) => theirs.some((r) => gid(r) === g)).map((g) => {
    const a = mine.filter((r) => gid(r) === g), b = theirs.filter((r) => gid(r) === g);
    const off = (rs: Row[]) => rs.filter((r) => r.pos !== "DST").reduce((x, r) => x + r.proj, 0);
    return { g, name: gname(a[0]), when: `${a[0].game.day} ${clock(a[0].game.time)}`, total: a[0].game.game_total, me: a.reduce((x, r) => x + r.proj, 0), them: b.reduce((x, r) => x + r.proj, 0),
      offMe: off(a), offThem: off(b), dst: [...a, ...b].some((r) => r.pos === "DST"), a, b };
  }).sort((x, y) => y.me + y.them - (x.me + x.them));
  // two or more of your starters from one offense: their weeks rise and fall together
  const stacks = Object.entries(mine.filter((r) => r.pos !== "DST" && r.pos !== "K").reduce<Record<string, Row[]>>((acc, r) => { (acc[r.team ?? "?"] ??= []).push(r); return acc; }, {}))
    .filter(([, rs]) => rs.length > 1);

  const line = (r: Row, who: "me" | "them" | "res") => {
    const l = lp.get(r.id), started = l && (l.state === "in" || l.state === "post");
    return (
    <button key={`${who}${r.id}`} type="button" className={`gl ${who} ${started ? l!.state : ""} ${selected === r.id ? "sel" : ""}`} onClick={() => who !== "them" && onSelect(r.id)} onDoubleClick={() => who !== "them" && openFile(r.id)} disabled={who === "them"}
      title={started ? `${l!.state === "post" ? "Final" : "So far"}: ${f1(l!.pts)} points. Projected ${f1(r.proj)} before kickoff.` : undefined}>
      <span className="nm">{lastName(r.name)}</span><span className="pt num">{f1(started ? l!.pts : r.proj)}</span>
      <span className="gm">{started && <span className="pj">{l!.state === "post" ? "final" : "so far"} · proj {f1(r.proj)} · </span>}{who === "res" ? "bench · " : r.slot ? `${r.slot} · ` : ""}{r.team} {r.game.home ? "vs" : "@"} {r.game.opp}</span>
      {r.game.implied_total != null && <span className="gm vegas" title={`Vegas: ${r.team} implied ${r.game.implied_total} of a ${r.game.game_total} total`}>team total {r.game.implied_total}</span>}
    </button>
    );
  };

  return (
    <div className="gameday">
      <header className="gd-hd">
        <h2>Week {tw.week} kickoffs</h2>
        <p>All times ET. Columns are kickoff windows; a player locks when his game starts. Late swaps are healthy bench players you can still move in up to their kickoff.</p>
      </header>
      {(shared.length > 0 || stacks.length > 0) && (
        <div className="gd-notes">
          {shared.map((x) => (
            <div key={x.g} className="gd-note">
              <b>{x.name}</b> <span className="dim">{x.when}{x.total != null ? ` · total ${x.total}` : ""}</span>
              <span className="who"><i className="me">you {f1(x.me)}</i> <i className="them">them {f1(x.them)}</i></span>
              <span className="dim">{x.a.map((r) => lastName(r.name)).join(", ")} vs {x.b.map((r) => lastName(r.name)).join(", ")}. {x.offMe > x.offThem + 3 ? "A high-scoring game helps you more than them." : x.offThem > x.offMe + 3 ? "A high-scoring game helps them more than you." : "A shootout lifts both sides about equally."}{x.dst ? " (A defense here moves the other way.)" : ""}</span>
            </div>
          ))}
          {stacks.map(([t, rs]) => (
            <div key={t} className="gd-note">
              <b>{t} stack</b> <span className="dim">{rs.map((r) => `${lastName(r.name)} ${f1(r.proj)}`).join(" + ")}</span>
              <span className="dim">Same offense, so their weeks tend to rise and fall together: more upside and more downside than the projections alone suggest.</span>
            </div>
          ))}
        </div>
      )}
      <div className="tt" style={{ gridTemplateColumns: `repeat(${wins.length}, minmax(130px, 1fr))` }}>
        {wins.map((w, i) => {
          const g = [...mine, ...theirs].find((r) => key(r.game) === w)?.game ?? reserves.find((p) => key(p.game!) === w)!.game!;
          const m = mine.filter((r) => key(r.game) === w), t = theirs.filter((r) => key(r.game) === w);
          const res = reserves.filter((p) => key(p.game!) === w);
          const sts = [...m, ...t, ...res].map((r) => lp.get(r.id)?.state).filter((x): x is LivePlayer["state"] => !!x);
          const badge = sts.some((x) => x === "in") ? "live" : sts.length && sts.every((x) => x === "post") ? "final" : null;
          return (
            <section key={w} className={`win ${i === 0 ? "first" : ""}`}>
              <div className="when"><span className="d">{g.day} {g.date?.slice(5).replace("-", "/")}{badge && <i className={`wst ${badge}`}>{badge === "live" ? "Live" : "Final"}</i>}</span><span className="t num">{clock(g.time)}</span>
                <span className="gs">{(() => { const gs = new Map<string, number | null>(); for (const r of [...m, ...t, ...res.map((p) => ({ team: p.team, game: p.game! }))]) { const nm = r.game.home ? `${r.game.opp}@${r.team}` : `${r.team}@${r.game.opp}`; gs.set(nm, r.game.game_total ?? null); } return [...gs].map(([nm, tot]) => <span key={nm}>{nm}{tot != null ? <em> {tot}</em> : null}</span>); })()}</span></div>
              {m.length > 0 && <><div className="lane">You · {f1(m.reduce((a, r) => a + r.proj, 0))}</div>{m.map((r) => line(r, "me"))}</>}
              {t.length > 0 && <><div className="lane">{teamShort(tw.matchup?.opp_name ?? "", [])} · {f1(t.reduce((a, r) => a + r.proj, 0))}</div>{t.map((r) => line(r, "them"))}</>}
              {res.length > 0 && <><div className="lane">Late swap</div>{res.map((p) => <div key={p.id} className="swapbox">{line({ id: p.id, name: p.name, pos: p.pos, team: p.team, proj: p.espn_week ?? 0, game: p.game! }, "res")}</div>)}</>}
            </section>
          );
        })}
      </div>
      <figure className="cum">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Projected points kicked off by each window">
          {[0, Math.round(mx / 2), Math.round(mx)].map((v) => <g key={v}><line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className="grid" /><text x={L - 6} y={y(v) + 4} textAnchor="end">{v}</text></g>)}
          <path d={path(1)} className="them-line" />
          <path d={path(0)} className="me-line" />
          {cum.map((c, i) => <circle key={i} cx={x(i)} cy={y(c[0])} r={4} className="me-dot" />)}
          {(() => { // keep the two end labels apart when the totals are close
            const a = y(cm) + 4, b = y(ct) + 4, gap = 13;
            const [ya, yb] = Math.abs(a - b) >= gap ? [a, b] : cm >= ct ? [Math.min(a, b) - gap / 2, Math.min(a, b) + gap / 2] : [Math.min(a, b) + gap / 2, Math.min(a, b) - gap / 2];
            return <><text x={x(cum.length - 1) + 8} y={ya} className="me-t">{f1(cm)} you</text><text x={x(cum.length - 1) + 8} y={yb} className="them-t">{f1(ct)} them</text></>;
          })()}
          {wins.map((w, i) => { const g = [...mine, ...theirs].find((r) => key(r.game) === w)?.game; return g ? <text key={w} x={x(i)} y={H - 3} textAnchor="middle">{g.day} {clock(g.time)}</text> : null; })}
          {half >= 0 && <><line x1={x(half)} x2={x(half)} y1={T - 6} y2={y(cum[half][0]) - 5} className="half" /><text x={x(half) + 6} y={T - 8} className="me-t">half your points kicked off by here</text></>}
        </svg>
        <figcaption><span className="k me" />You, projected points kicked off so far <span className="k them" />{tw.matchup?.opp_name}</figcaption>
      </figure>
    </div>
  );
}

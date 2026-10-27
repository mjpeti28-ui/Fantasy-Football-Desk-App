import { useEffect, useState } from "react";
import type { FinderOpts, FinderPlayer, FinderRow, FinderState, LeagueLive, PowerTeam } from "../../types";
import { Ev } from "../../components/Analytics";
import { pct, signed, tightName } from "../../format";
import { tradeApi } from "../../data";
import type { Deal } from "./Trades";

const POS = ["QB", "RB", "WR", "TE"];
const SHAPES = ["1-1", "2-1", "1-2", "2-2", "3-1", "3-2"];
const SLOW = new Set(["2-2", "3-1", "3-2"]);
const DEFAULT: FinderOpts = { shapes: ["1-1", "2-1", "1-2"], keep: [], must_give: [], must_get: [], get_pos: POS, partners: null,
  horizon: "ros", desk_weight: 0.5, min_gain: 0, min_their: 0, their_view: "espn", sort: "gain" };

const toggle = <T,>(xs: T[], x: T) => (xs.includes(x) ? xs.filter((y) => y !== x) : [...xs, x]);
const tone = (x: number) => (x >= 10 ? "up" : x <= -10 ? "down" : "");

/* FantasyPros-style trade finder: filters on who moves, and a valuation you set (how much to trust
   Desk vs ESPN, which weeks count, how much the partner must gain and on whose numbers). */
export function Finder({ me, teams, live, engineUp, setDeal }: { me: number; teams: PowerTeam[]; live: LeagueLive | null; engineUp: boolean; setDeal: (d: Deal) => void }) {
  const [o, setO] = useState<FinderOpts>(DEFAULT);
  const [st, setSt] = useState<FinderState | null>(null);
  const set = (x: Partial<FinderOpts>) => setO((p) => ({ ...p, ...x }));
  useEffect(() => { if (engineUp) tradeApi.finder().then(setSt, () => undefined); }, [engineUp]);
  useEffect(() => {
    if (st?.state !== "running") return;
    const t = window.setInterval(() => tradeApi.finder().then(setSt, () => undefined), 2000);
    return () => window.clearInterval(t);
  }, [st?.state]);
  const mine = (live?.rosters[String(me)] ?? []).filter((r) => POS.includes(r.pos) && !r.ir);
  const others = teams.filter((t) => t.id !== me);
  const name = (t: number) => teams.find((x) => x.id === t)?.name ?? `Team ${t}`;
  const theirs = others.flatMap((t) => (live?.rosters[String(t.id)] ?? []).filter((r) => POS.includes(r.pos)).map((r) => ({ ...r, owner: t.id })))
    .sort((a, b) => (b.est_ppg ?? 0) - (a.est_ppg ?? 0));
  // my players cycle: tradeable -> must include -> untouchable
  const mode = (id: number) => (o.must_give.includes(id) ? "must" : o.keep.includes(id) ? "keep" : "");
  const cycle = (id: number) => {
    const m = mode(id);
    set({ must_give: m === "" ? [...o.must_give, id] : o.must_give.filter((x) => x !== id), keep: m === "must" ? [...o.keep, id] : o.keep.filter((x) => x !== id) });
  };
  const run = async () => setSt(await tradeApi.startFinder(o));
  const secs = st?.state === "running" && st.started ? Math.round(Date.now() / 1000 - st.started) : 0;
  const w = Math.round(o.desk_weight * 100);
  const slow = o.shapes.some((s) => SLOW.has(s));
  const ppl = (xs: FinderPlayer[]) => xs.map((x) => <span key={x.id} className="fp"><b title={x.name}>{tightName(x.name, 18)}</b> <small>{x.pos} · {Math.round(x.ros)}</small></span>);

  return (
    <section className="calls finder" aria-label="Trade finder">
      <h3>Trade finder <Ev l="descriptive" tip="Ranks deals by the valuation you set. Gains are change in your starting-lineup points over the weeks you pick. Under ~10 points is noise in the backtest." /><span>set who moves and how a deal is valued, then search</span></h3>
      <div className="fz">
        <div className="fg wide"><label>Trade away <small>click: tradeable → must include → untouchable</small></label>
          <div className="chips">{mine.map((r) => <button key={r.id} type="button" className={`ch ${mode(r.id)}`} onClick={() => cycle(r.id)} title={`${r.name} (${r.pos})`}>{tightName(r.name, 14)} <small>{mode(r.id) === "keep" ? "keep" : mode(r.id) === "must" ? "must go" : r.pos}</small></button>)}</div></div>
        <div className="fg"><label>Trade for</label>
          <div className="chips">{POS.map((p) => <button key={p} type="button" className={`ch ${o.get_pos.includes(p) ? "on" : ""}`} onClick={() => set({ get_pos: toggle(o.get_pos, p) })}>{p}</button>)}</div>
          <select value={o.must_get[0] ?? ""} onChange={(e) => set({ must_get: e.target.value ? [Number(e.target.value)] : [] })}>
            <option value="">Any player</option>
            {theirs.map((r) => <option key={r.id} value={r.id}>{r.name} ({r.pos}, {name(r.owner)})</option>)}
          </select></div>
        <div className="fg"><label>Trade with</label>
          <div className="chips">
            <button type="button" className={`ch ${o.partners == null ? "on" : ""}`} onClick={() => set({ partners: null })}>All</button>
            {others.map((t) => <button key={t.id} type="button" className={`ch ${o.partners?.includes(t.id) ? "on" : ""}`}
              onClick={() => { const n = toggle(o.partners ?? [], t.id); set({ partners: n.length ? n : null }); }}>{tightName(t.name, 14)}</button>)}</div></div>
        <div className="fg"><label>Trade type <small>you send-you get{slow ? ", wide shapes take minutes" : ""}</small></label>
          <div className="chips">{SHAPES.map((s) => <button key={s} type="button" className={`ch ${o.shapes.includes(s) ? "on" : ""}`} onClick={() => set({ shapes: toggle(o.shapes, s) })}>{s}</button>)}</div></div>
      </div>
      <div className="fz val">
        <div className="fg wide"><label>Your gain = {w}% Desk + {100 - w}% ESPN <Ev l="weak" tip="Desk's rest-of-season level only barely beats ESPN's (error 3.67 vs 3.74 ppg, next 4 weeks), and in the weekly backtest Desk overweighted its gap with ESPN about 4x. 50% is a hedge, not a fitted number." /></label>
          <input type="range" min={0} max={100} step={10} value={w} onChange={(e) => set({ desk_weight: Number(e.target.value) / 100 })} />
          <small className="dim">0% = what ESPN's app shows you · 100% = our model alone</small></div>
        <div className="fg"><label>Weeks that count</label>
          <select value={o.horizon} onChange={(e) => set({ horizon: e.target.value as FinderOpts["horizon"] })}>
            <option value="ros">Rest of season</option><option value="reg">Regular season only</option><option value="playoffs">Playoff weeks only</option></select></div>
        <div className="fg"><label>Your gain at least</label><input type="number" step={5} value={o.min_gain} onChange={(e) => set({ min_gain: Number(e.target.value) })} /></div>
        <div className="fg"><label>Their gain at least <Ev l="weak" tip="Accepted offers in this league have looked fair on ESPN's numbers, but that's ~20 answered proposals: an assumption, not a fitted acceptance model." /></label>
          <span className="row"><input type="number" step={5} value={o.min_their} onChange={(e) => set({ min_their: Number(e.target.value) })} />
            <select value={o.their_view} onChange={(e) => set({ their_view: e.target.value as FinderOpts["their_view"] })}>
              <option value="espn">on ESPN</option><option value="desk">on Desk</option><option value="both">on both</option></select></span></div>
        <div className="fg"><label>Sort by</label>
          <select value={o.sort} onChange={(e) => set({ sort: e.target.value as FinderOpts["sort"] })}>
            <option value="gain">Your gain</option><option value="their">Their gain</option><option value="fair">Fairest win (smaller of the two)</option></select></div>
        <div className="fg go"><button className="tbtn pri" type="button" disabled={!engineUp || st?.state === "running" || !o.shapes.length || !o.get_pos.length} onClick={run}>
          {st?.state === "running" ? `Searching… ${secs}s` : o.partners == null && !o.must_get.length && !o.must_give.length ? "Find trades (~2 min)" : "Find trades"}</button>
          <button className="tbtn" type="button" onClick={() => setO(DEFAULT)}>Reset</button></div>
      </div>
      {st?.state === "failed" && <p className="tr-off">Search failed: {st.error}</p>}
      {st?.state === "done" && <p className="tr-empty">{st.candidates} packages scored{st.truncated ? " (capped: the closest swaps on raw points)" : ""}, {st.rows.length} pass your filters.
        {st.opts && st.opts.desk_weight !== o.desk_weight && " Weights changed since this search: run it again."}</p>}
      {(st?.rows ?? []).length > 0 && (
        <div className="ft-wrap"><table className="ft">
          <thead><tr><th>#</th><th>You give</th><th>You get</th><th>With</th>
            <th title="Blend you set">Your gain</th><th title="Desk / ESPN">Desk · ESPN</th><th title="Desk minus ESPN: how much of this only exists on our numbers">Gap</th>
            <th title="Partner's gain on ESPN (what their app shows) / on Desk">Them: ESPN · Desk</th>
            <th title="Desk's gain split: the players' own rest-of-season points vs lineup fit (bench, byes, drops)">Players · fit</th><th>Trust</th><th /></tr></thead>
          <tbody>{st!.rows.map((r: FinderRow, i) => (
            <tr key={i} title={r.why}>
              <td className="num">{i + 1}</td><td>{ppl(r.give)}</td><td>{ppl(r.get)}</td><td>{tightName(r.with, 16)}{r.you_drop && <small className="dim d">you drop {r.you_drop}</small>}</td>
              <td className={`num big ${tone(r.your_gain)}`}>{signed(r.your_gain)}</td>
              <td className="num">{signed(r.desk_gain)} · {signed(r.espn_gain)}</td>
              <td className={`num ${Math.abs(r.desk_gain - r.espn_gain) >= 15 ? "warn" : ""}`}>{signed(r.desk_gain - r.espn_gain)}</td>
              <td className="num">{signed(r.their_espn)} · {signed(r.their_desk)}</td>
              <td className="num">{signed(r.players)} · {signed(r.fit)}</td>
              <td><span className={`conf ${r.trust === "noise" ? "noise" : r.trust === "thin edge" ? "thin" : "yes"}`}>{r.trust}</span> <small className="dim">{pct(r.p_right)}</small></td>
              <td><button className="tbtn" type="button" onClick={() => setDeal({ partner: r.partner, give: r.give_ids, get: r.get_ids })}>Open desk →</button></td>
            </tr>))}</tbody>
        </table></div>
      )}
      <p className="tr-empty">Gains are points over the weeks you picked. Gap = Desk minus ESPN for you: a deal that only wins on Desk is a bet on our model, which has beaten ESPN only narrowly. "Trust" and the percentage (how often a gain this size had the right sign, 2021-25) were calibrated on Desk's gains, so they're rough for blended numbers.</p>
    </section>
  );
}

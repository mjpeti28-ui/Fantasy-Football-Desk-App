import { useEffect, useState } from "react";
import type { FinderOpts, FinderPlayer, FinderRow, FinderState, LeagueLive, PowerTeam } from "../../types";
import { signed, tightName } from "../../format";
import { tradeApi } from "../../data";
import type { Deal } from "./Trades";

const POS = ["QB", "RB", "WR", "TE"];
const POS_VAR: Record<string, string> = { QB: "var(--qb)", RB: "var(--rb)", WR: "var(--wr)", TE: "var(--te)" };
const SHAPES = ["1-1", "2-1", "1-2", "2-2", "3-1", "3-2"];
const SLOW = new Set(["2-2", "3-1", "3-2"]);
// search depth presets: sims in the cheap first pass, packages scored at most, sims when rescoring the shortlist
const DEPTH = { quick: { sims: 10, max_cands: 3000, rescore_sims: 150 }, normal: { sims: 20, max_cands: 12000, rescore_sims: 300 },
  deep: { sims: 40, max_cands: 40000, rescore_sims: 600 } };
type Depth = keyof typeof DEPTH;
const DEFAULT: FinderOpts = { shapes: ["1-1", "2-1", "1-2"], keep: [], must_give: [], must_get: [], get_pos: POS, partners: null,
  horizon: "ros", desk_weight: 0.5, min_gain: 0, min_their: 0, their_view: "espn", sort: "gain", top: 25, ...DEPTH.normal };

const toggle = <T,>(xs: T[], x: T) => (xs.includes(x) ? xs.filter((y) => y !== x) : [...xs, x]);
const n0 = (x: number) => { const r = Math.round(x); return `${r > 0 ? "+" : r < 0 ? "−" : ""}${Math.abs(r)}`; };

/** How the two views read a deal for you: agree, or most of the gain lives in one of them. */
function read(r: FinderRow, po: boolean): { t: string; tone: string; tip: string } {
  // playoff-week gains are ~1/3 the size of rest-of-season ones, so the cutoffs scale down
  const gap = r.desk_gain - r.espn_gain, small = po ? 6 : 10, big = po ? 5 : 15;
  if (r.your_gain < small) return { t: "Small edge", tone: "dim", tip: `Under ${small} points: in the 2021-25 backtest edges this size had the right sign only a bit more than half the time.` };
  if (gap >= big) return { t: "Mostly our model", tone: "warn", tip: `Desk sees ${n0(gap)} more than ESPN. A bet that our model is right where ESPN isn't; Desk has beaten ESPN only narrowly.` };
  if (gap <= -big) return { t: "Mostly ESPN", tone: "mid", tip: `ESPN sees ${n0(-gap)} more than Desk. Looks great on the app; our model is less sold.` };
  return { t: "Both agree", tone: "good", tip: `Desk and ESPN both have you gaining ${small}+ points, within ${big} of each other.` };
}

/** Partner's side on what their app shows: our only (weak) proxy for whether they'd accept. */
function theirRead(x: number, po: boolean): { t: string; tone: string } {
  if (x >= (po ? 2 : 5)) return { t: "fair on their app", tone: "good" };
  if (x >= 0) return { t: "barely fair", tone: "mid" };
  return { t: "looks bad to them", tone: "warn" };
}

function Info({ tip }: { tip: string }) {
  return <i className="fi" title={tip} aria-label={tip}>?</i>;
}

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
  const depth = (Object.keys(DEPTH) as Depth[]).find((k) => DEPTH[k].sims === o.sims && DEPTH[k].max_cands === o.max_cands) ?? "normal";
  const wide = o.partners == null && !o.must_get.length && !o.must_give.length;
  const est = wide ? { quick: "~30 s", normal: "~2 min", deep: "5+ min" }[depth] : "";
  const HZ = { ros: "rest of season", reg: "regular season", playoffs: "playoff weeks", title: "season, weighted by title odds (in playoff points)" };
  const weeks = HZ[o.horizon];
  const ppl = (xs: FinderPlayer[]) => xs.map((x) => (
    <span key={x.id} className="fp"><i className="pd" style={{ background: POS_VAR[x.pos] }} /><b title={x.name}>{tightName(x.name, 18)}</b><small>{x.pos} · {Math.round(x.ros)} pts</small></span>));
  const stale = st?.opts && (st.opts.desk_weight !== o.desk_weight || st.opts.horizon !== o.horizon);

  return (
    <section className="finder" aria-label="Trade finder">
      <header className="f-hd">
        <h3>Trade finder</h3>
        <p>Pick who can move and how a deal should be valued. Every deal is scored twice: on <b>Desk</b> (our model) and on <b>ESPN</b> (what your partner's app shows).</p>
      </header>

      <div className="f-card">
        <h4>Who's in the deal</h4>
        <div className="fg">
          <label>You'd trade away <small>click a player: open → <b className="lg-must">must go</b> → <s>untouchable</s></small></label>
          <div className="chips">{mine.map((r) => (
            <button key={r.id} type="button" className={`ch ${mode(r.id)}`} onClick={() => cycle(r.id)} title={`${r.name} (${r.pos}): ${mode(r.id) === "must" ? "every deal includes him" : mode(r.id) === "keep" ? "never traded" : "may be traded"}`}>
              <i className="pd" style={{ background: POS_VAR[r.pos] }} />{tightName(r.name, 14)}</button>))}</div>
        </div>
        <div className="f-row">
          <div className="fg"><label>You want</label>
            <div className="chips">{POS.map((p) => <button key={p} type="button" className={`ch ${o.get_pos.includes(p) ? "on" : ""}`} onClick={() => set({ get_pos: toggle(o.get_pos, p) })}><i className="pd" style={{ background: POS_VAR[p] }} />{p}</button>)}</div>
            <select className="full" value={o.must_get[0] ?? ""} onChange={(e) => set({ must_get: e.target.value ? [Number(e.target.value)] : [] })}>
              <option value="">Any player at those positions</option>
              {theirs.map((r) => <option key={r.id} value={r.id}>{r.name} ({r.pos}, {name(r.owner)})</option>)}
            </select></div>
          <div className="fg"><label>Deal shape <small>players you send – players you get</small></label>
            <div className="chips">{SHAPES.map((s) => <button key={s} type="button" className={`ch mono ${o.shapes.includes(s) ? "on" : ""}`} onClick={() => set({ shapes: toggle(o.shapes, s) })}>{s.replace("-", " for ")}</button>)}</div>
            {slow && <small className="dim">2-for-2 and 3-player shapes make the search a lot slower.</small>}</div>
        </div>
        <div className="fg"><label>With</label>
          <div className="chips">
            <button type="button" className={`ch ${o.partners == null ? "on" : ""}`} onClick={() => set({ partners: null })}>Any team</button>
            {others.map((t) => <button key={t.id} type="button" className={`ch ${o.partners?.includes(t.id) ? "on" : ""}`}
              onClick={() => { const n = toggle(o.partners ?? [], t.id); set({ partners: n.length ? n : null }); }}>{t.name}</button>)}</div></div>
      </div>

      <div className="f-card">
        <h4>How to value it</h4>
        <div className="fg">
          <label>Whose numbers count for you <Info tip="Your gain = this blend of Desk and ESPN. Desk's rest-of-season level beats ESPN's only narrowly (3.67 vs 3.74 ppg error, next 4 weeks), and in the weekly backtest Desk overweighted its disagreements with ESPN about 4x. 50/50 is a hedge (opinion), not a fitted number." /></label>
          <div className="blend">
            <span className={w <= 50 ? "on" : ""}>ESPN {100 - w}%</span>
            <input type="range" min={0} max={100} step={10} value={w} onChange={(e) => set({ desk_weight: Number(e.target.value) / 100 })} aria-label="Desk weight" />
            <span className={w >= 50 ? "on" : ""}>Desk {w}%</span>
          </div>
        </div>
        <div className="f-row four">
          <div className="fg"><label>Weeks that count</label>
            <select value={o.horizon} onChange={(e) => set({ horizon: e.target.value as FinderOpts["horizon"] })}>
              <option value="ros">Rest of season</option><option value="reg">Regular season only</option><option value="playoffs">Playoff weeks only</option>
              <option value="title" title="Each team's regular-season points count by how much they move its title odds: ~0 for a team that's already in, more for a bubble team. Gains in playoff points.">Weighted by title odds</option></select></div>
          <div className="fg"><label>You gain at least</label>
            <span className="row"><input type="number" step={5} value={o.min_gain} onChange={(e) => set({ min_gain: Number(e.target.value) })} /><small>pts</small></span></div>
          <div className="fg"><label>They gain at least <Info tip="Accepted offers in this league have looked fair on ESPN's numbers, which is what their app shows. That's from ~20 answered proposals: an assumption, not a fitted acceptance model." /></label>
            <span className="row"><input type="number" step={5} value={o.min_their} onChange={(e) => set({ min_their: Number(e.target.value) })} /><small>pts</small>
              <select value={o.their_view} onChange={(e) => set({ their_view: e.target.value as FinderOpts["their_view"] })}>
                <option value="espn">on ESPN</option><option value="desk">on Desk</option><option value="both">on both</option></select></span></div>
          <div className="fg"><label>Sort by</label>
            <select value={o.sort} onChange={(e) => set({ sort: e.target.value as FinderOpts["sort"] })}>
              <option value="gain">Your gain</option><option value="their">Their gain</option><option value="fair">Fairest win (smaller of the two)</option></select></div>
        </div>
        <div className="f-go">
          <div className="fg"><label>Search depth <Info tip="Quick: 10 sims per deal, up to 3,000 deals (keeps the most even swaps if there are more). Normal: 20 sims, 12,000 deals. Deep: 40 sims, 40,000 deals, shortlist rescored at 600 sims. Deeper misses fewer good deals and takes longer." /></label>
            <span className="row"><span className="seg">{(Object.keys(DEPTH) as Depth[]).map((k) => <button key={k} type="button" className={depth === k ? "on" : ""} onClick={() => set(DEPTH[k])}>{k[0].toUpperCase() + k.slice(1)}</button>)}</span>
              <small>show</small><input type="number" min={1} max={60} value={o.top} onChange={(e) => set({ top: Math.max(1, Math.min(60, Number(e.target.value) || 25)) })} /><small>deals</small></span></div>
          <span className="sp" />
          <button className="tbtn" type="button" onClick={() => setO(DEFAULT)}>Reset</button>
          <button className="tbtn pri big" type="button" disabled={!engineUp || st?.state === "running" || !o.shapes.length || !o.get_pos.length} onClick={run}>
            {st?.state === "running" ? `Searching… ${secs}s` : est ? `Find trades · ${est}` : "Find trades"}</button>
        </div>
      </div>

      {st?.state === "failed" && <p className="tr-off">Search failed: {st.error}</p>}
      {st?.state === "done" && (
        <p className="f-sum"><b>{st.rows.length}</b> deals pass your filters out of {st.candidates?.toLocaleString()} scored{st.truncated ? " (hit the depth cap: try a deeper search)" : ""}.
          {stale && <span className="warn"> You changed the valuation since this search: run it again.</span>}</p>
      )}
      {(st?.rows ?? []).length > 0 && (<>
        <div className="f-key"><span><b>Your gain</b> = {st?.opts ? Math.round(st.opts.desk_weight * 100) : w}% Desk + {st?.opts ? 100 - Math.round(st.opts.desk_weight * 100) : 100 - w}% ESPN, starting-lineup points over the {st?.opts ? HZ[st.opts.horizon] : weeks}</span>
          <span><b>Odds</b>: how often a gain this size went the predicted way, 2021-25</span><span>Hover a row for the players' notes</span></div>
        <div className="ft-wrap"><table className="ft">
          <thead><tr><th>#</th><th>You give</th><th>You get</th>
            <th className="r">Your gain</th><th title="Do Desk and ESPN agree on your gain?">Desk vs ESPN</th><th title="Partner's gain on ESPN (their app) and on Desk">Their side</th><th className="r" title="How often a gain this size went the predicted way, 2021-25">Odds</th><th /></tr></thead>
          <tbody>{st!.rows.map((r: FinderRow, i) => {
            const po = st?.opts?.horizon === "playoffs" || st?.opts?.horizon === "title";
            const rd = read(r, po), th = theirRead(r.their_espn, po);
            return (
              <tr key={i} title={r.why}>
                <td className="rk">{i + 1}</td>
                <td>{ppl(r.give)}</td>
                <td>{ppl(r.get)}<small className="sub wrap">from {r.with}{r.you_drop && ` · then drop ${r.you_drop}`}</small></td>
                <td className="r"><span className={`gain ${r.your_gain >= 10 ? "up" : r.your_gain < 0 ? "down" : ""}`}>{signed(r.your_gain)}</span>
                  <small className="sub">Desk {n0(r.desk_gain)} · ESPN {n0(r.espn_gain)}</small></td>
                <td><span className={`tag ${rd.tone}`} title={rd.tip}>{rd.t}</span></td>
                <td><span className={`tag ${th.tone}`}>{th.t}</span><small className="sub">ESPN {n0(r.their_espn)} · Desk {n0(r.their_desk)}</small></td>
                <td className="r"><span className={`odds ${r.p_right >= 0.65 ? "good" : r.p_right >= 0.58 ? "mid" : "dim"}`}>{Math.round(r.p_right * 100)}%</span></td>
                <td className="r"><button className="tbtn" type="button" onClick={() => setDeal({ partner: r.partner, give: r.give_ids, get: r.get_ids })}>Open →</button></td>
              </tr>);
          })}</tbody>
        </table></div>
        <p className="tr-empty">Odds were calibrated on Desk's gains alone, so they're rough for a blend. "Mostly our model" deals are the ones to double-check: Desk has beaten ESPN only narrowly, and ESPN is what your partner will see.</p>
      </>)}
    </section>
  );
}

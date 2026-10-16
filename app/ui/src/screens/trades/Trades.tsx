import { useEffect, useMemo, useRef, useState } from "react";
import type { AFlag, APlayer, Analytics, IdeasState, JDraft, LeagueAnalysis, LeagueLive, PowerTeam, RosterRow, TradeEval, TradeHistory, TradeOdds } from "../../types";
import { Ev, EvFor, FlagChip } from "../../components/Analytics";
import { f1, pct, signed } from "../../format";
import { tradeApi } from "../../data";
import { useFile } from "../../components/FileContext";
import { Face } from "../../components/bits";
import { type TeamColor, tilePalette } from "../../teamColor";
import "./trades.css";

export interface Deal { partner: number | null; give: number[]; get: number[] }

interface Props {
  me: number; teams: PowerTeam[]; live: LeagueLive | null; engineUp: boolean; teamColors?: Record<string, TeamColor>;
  history: Record<string, TradeHistory>; deal: Deal; setDeal: (d: Deal) => void; onCopy?: (md: string) => void; week?: number; onLog?: (d: JDraft) => void;
  /** season analytics: inline flags on roster rows, the deal's season-so-far check, the call sheet's disagreement board */
  an?: Analytics | null;
}

/** The one flag worth showing on a roster row: running hot/cold vs usage first, then a model value move. */
const rowFlag = (fs?: AFlag[]) => fs?.find((f) => f.key === "hot" || f.key === "cold") ?? fs?.find((f) => f.key.startsWith("value"));

const POS_VAR: Record<string, string> = { QB: "var(--qb)", RB: "var(--rb)", WR: "var(--wr)", TE: "var(--te)", K: "var(--k)", DST: "var(--dst)" };
const dark = () => typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches;

export function historyLabel(h?: TradeHistory) {
  if (!h || !h.sent) return { text: "no offers from you yet", tone: "" };
  if (h.accepted) return { text: `accepted ${h.accepted} of ${h.sent}`, tone: "good" };
  if (h.declined) return { text: `0 of ${h.sent} · declines`, tone: "bad" };
  return { text: `0 of ${h.sent} · never answers`, tone: "bad" };
}

/* partner chips with acceptance history and trade-fit marks */
function Partners({ me, teams, history, deal, setDeal, fits }: Props & { fits: Set<number> }) {
  return (
    <div className="partners" role="radiogroup" aria-label="Trade partner">
      {teams.filter((t) => t.id !== me).map((t) => {
        const h = historyLabel(history[t.id]);
        return (
          <button key={t.id} type="button" role="radio" aria-checked={deal.partner === t.id} className={`pc ${deal.partner === t.id ? "on" : ""}`}
            onClick={() => setDeal({ partner: t.id, give: deal.partner === t.id ? deal.give : [], get: [] })}>
            <b>{t.name}</b>
            <span className={`h ${h.tone}`}>{h.text}</span>
            {fits.has(t.id) && <span className="fit">fit</span>}
          </button>
        );
      })}
    </div>
  );
}

function Roster({ rows, picked, onToggle, title, side, colors, flags }: { rows: RosterRow[]; picked: number[]; onToggle: (id: number) => void; title: string; side: "me" | "them"; colors?: Record<string, TeamColor>; flags?: Map<number, AFlag[]> }) {
  const dk = dark();
  const { open, compare } = useFile();
  return (
    <section className={`roster ${side}`}>
      <h5>{title}</h5>
      <div className="rlist">
        {rows.filter((r) => !r.ir).map((r) => {
          const pal = tilePalette(r.team ?? null, colors, dk), on = picked.includes(r.id);
          const st = pal ? { ["--tbg" as string]: pal.bg, ["--thi" as string]: pal.hi, ["--tlo" as string]: pal.lo, ["--tac" as string]: pal.accent } : {};
          return (
            <button key={r.id} type="button" className={`trow ${pal ? "team" : ""} ${on ? "on" : ""}`} style={st} onClick={(e) => { if (e.metaKey && r.pos !== "K" && r.pos !== "DST") compare(r.id); else onToggle(r.id); }} onDoubleClick={() => open(r.id, rows.filter((x) => !x.ir).map((x) => x.id))} aria-pressed={on}
              title={`${r.name}: ${f1(r.est_ppg)} ppg rest of season, plays ${pct(r.avail)}`}>
              <span className="fw"><Face p={{ headshot: r.headshot, pos: r.pos, name: r.name }} className="round" />
                {(() => { const fl = rowFlag(flags?.get(r.id)); return fl ? <i className={`tfb ${fl.tone} ${fl.label}`} title={`${fl.short}: ${fl.text}`} aria-label={fl.short}>{fl.tone === "up" ? "↑" : "↓"}</i> : null; })()}</span>
              <span className="nm">{r.name}</span>
              <i className={`pchip ${r.pos === "WR" ? "wr" : ""}`} style={{ background: POS_VAR[r.pos] }}>{r.pos === "DST" ? "D/ST" : r.pos}</i>
              <span className="v num">{f1(r.est_ppg)}</span>
              <span className="mk" aria-hidden="true">{on ? (side === "me" ? "→" : "←") : "+"}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

/* gains on one axis per view; the band around zero is the "noise" zone from the backtest */
function GainChart({ ev, me, partner, pname }: { ev: TradeEval; me: number; partner: number; pname: string }) {
  const rows = [
    { view: "Our model", who: "You", v: ev.model.teams[me].delta_total_calibrated, cls: "me" },
    { view: "", who: pname, v: ev.model.teams[partner].delta_total_calibrated, cls: "them" },
    { view: "ESPN's numbers", who: "You", v: ev.espn.teams[me].delta_total, cls: "me" },
    { view: "", who: pname, v: ev.espn.teams[partner].delta_total, cls: "them" },
  ];
  const span = Math.max(30, ...rows.map((r) => Math.abs(r.v))) * 1.1;
  const W = 420, L = 150, R = 44, H = 26 * rows.length + 46, x = (v: number) => L + ((W - L - R) * (v + span)) / (2 * span);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="gain" role="img" aria-label="Rest-of-season points gained by each side">
      <rect x={x(-10)} y={4} width={x(10) - x(-10)} height={H - 30} className="noise" />
      <text x={(x(-10) + x(10)) / 2} y={H - 14} textAnchor="middle" className="nz">noise</text>
      <line x1={x(0)} x2={x(0)} y1={4} y2={H - 26} className="zero" />
      {[-20, 20].filter((t) => Math.abs(t) < span).map((t) => <text key={t} x={x(t)} y={H - 14} textAnchor="middle" className="tk">{t > 0 ? `+${t}` : t}</text>)}
      {rows.map((r, i) => {
        const y = 12 + i * 26 + (i >= 2 ? 10 : 0), a = x(Math.min(0, r.v)), b = x(Math.max(0, r.v));
        return (
          <g key={i} className={r.cls}>
            {r.view && <text x={0} y={y + 13} className="view">{r.view}</text>}
            <text x={L - 8} y={y + 13} textAnchor="end" className="who">{r.who.length > 18 ? r.who.slice(0, 17) + "…" : r.who}</text>
            <rect x={a} y={y + 2} width={Math.max(2, b - a)} height={14} rx={3} className="bar" />
            <text x={r.v >= 0 ? b + 6 : a - 6} y={y + 13} textAnchor={r.v >= 0 ? "start" : "end"} className="val">{signed(r.v)}</text>
          </g>
        );
      })}
      <text x={W - R + 4} y={H - 14} className="tk">pts</text>
    </svg>
  );
}

export function Trades(p: Props) {
  const { me, teams, live, deal, setDeal, history, engineUp } = p;
  const [ev, setEv] = useState<TradeEval | null>(null);
  const [odds, setOdds] = useState<TradeOdds | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [oddsBusy, setOddsBusy] = useState(false);
  const seq = useRef(0);
  const partner = deal.partner;
  const pTeam = teams.find((t) => t.id === partner);
  const pname = pTeam?.name ?? "";
  const fits = useMemo(() => fitTeams(live?.analysis, me), [live, me]);
  const flags = useMemo(() => new Map((p.an?.players ?? []).map((x) => [x.id, x.flags])), [p.an]);

  useEffect(() => {
    setEv(null); setOdds(null); setErr(null);
    if (!engineUp || partner == null || !deal.give.length || !deal.get.length) return;
    const n = ++seq.current;
    const t = window.setTimeout(async () => {
      try {
        const e = await tradeApi.evaluate(partner, deal.give, deal.get);
        if (n !== seq.current) return;
        setEv(e); setOddsBusy(true);
        const o = await tradeApi.odds(partner, deal.give, deal.get);
        if (n === seq.current) setOdds(o);
      } catch (e) { if (n === seq.current) setErr(String(e)); }
      finally { if (n === seq.current) setOddsBusy(false); }
    }, 250);
    return () => window.clearTimeout(t);
  }, [engineUp, partner, deal.give.join(","), deal.get.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = (side: "give" | "get") => (id: number) => {
    const cur = deal[side];
    setDeal({ ...deal, [side]: cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id] });
  };
  const myRows = live?.rosters[me] ?? [], theirRows = partner != null ? live?.rosters[partner] ?? [] : [];
  const byId = new Map([...myRows, ...theirRows].map((r) => [r.id, r]));
  const m = ev?.model.teams[me], mt = partner != null ? ev?.model.teams[partner] : undefined, et = partner != null ? ev?.espn.teams[partner] : undefined;
  const h = historyLabel(partner != null ? history[partner] : undefined);

  const verdict = () => {
    if (!m || !mt || !et) return null;
    const mine = m.edge_label === "noise" ? `For you this is inside the noise: deals with an edge this small go your way about ${pct(m.p_direction_right)} of the time.`
      : `Good for you: ${signed(m.delta_total_calibrated)} rest-of-season points, right about ${pct(m.p_direction_right)} of the time at this size.`;
    const bad = m.delta_total_calibrated < 0 ? `It costs you ${f1(-m.delta_total_calibrated)} points by our model. ` : "";
    const theirs = et.delta_total >= 0 ? `On ESPN's numbers it looks fair or better to ${pname}, which is how the offers accepted in this league have looked.`
      : mt.delta_total_calibrated >= 0 ? `Our model says it helps ${pname}, but their app will show them losing ${f1(-et.delta_total)}. That's a hard sell.`
      : `It's a loss for ${pname} on both views (${signed(mt.delta_total_calibrated)} ours, ${signed(et.delta_total)} ESPN's). Expect a decline unless you add something.`;
    return <p className="verdict">{bad}{m.delta_total_calibrated >= 0 ? mine : ""} {theirs}</p>;
  };

  const markdown = () => {
    if (!m || !mt || !et || partner == null) return "";
    const o = odds?.teams;
    const L = [`## Trade draft: ${m.gives.join(", ")} for ${m.gets.join(", ")} with ${pname} (from Desk)`, "",
      "| | Our model (calibrated) | ESPN's numbers |", "|---|---|---|",
      `| You | ${signed(m.delta_total_calibrated)} | ${signed(ev!.espn.teams[me].delta_total)} |`,
      `| ${pname} | ${signed(mt.delta_total_calibrated)} | ${signed(et.delta_total)} |`, "",
      `Direction right at this edge: ${pct(m.p_direction_right)} (${m.edge_label}). Regular season / playoffs: ${signed(m.delta_regular_season)} / ${signed(m.delta_playoffs)}.`,
      m.would_drop.length ? `You'd drop: ${m.would_drop.join(", ")}.` : "",
      o ? `Season odds after: you top-2 ${pct(o[me].after.top2)} (${signed(100 * o[me].delta.top2)} pts), playoffs ${pct(o[me].after.playoffs)}; ${pname} top-2 ${pct(o[partner].after.top2)}, playoffs ${pct(o[partner].after.playoffs)}.` : "",
      `History with ${pname}: ${h.text} (2021–25).`, "", "Why:", ...Object.entries(m.why).map(([n, w]) => `- ${n}: ${w}`),
      ...(p.an ? ["", "Season so far (points per game / what usage usually produces / model rest of season; scoring over usage fades, ~9% carries):",
        ...[...deal.give, ...deal.get].map((id) => p.an!.players.find((x) => x.id === id)).filter((x): x is APlayer => !!x)
          .map((x) => `- ${x.name}: ${f1(x.ppg)} / ${f1(x.xfp)} / ${f1(x.est_ppg)}${x.flags.length ? ` (${x.flags.map((f) => f.short).join("; ")})` : ""}`)] : [])];
    return L.filter((x) => x !== "").join("\n");
  };

  const oddsRow = (t: number, label: string, cls: string) => {
    const d = odds?.teams[t]?.delta, a = odds?.teams[t]?.after, nz = (odds?.noise_pp ?? 1.1) / 100;
    const cell = (k: "top2" | "playoffs" | "seed1") => d && a ? (
      <span><b className="num">{pct(a[k])}</b> <em className={Math.abs(d[k]) < 2 * nz ? "flat" : d[k] > 0 ? "up" : "down"}>{Math.abs(d[k]) < 2 * nz ? "≈" : `${d[k] > 0 ? "+" : "−"}${Math.round(Math.abs(d[k]) * 100)}`}</em></span>) : <span className="dim">–</span>;
    const sm = d?.seed_med != null && a?.seed_med != null ? (
      <span><b className="num">{a.seed_med}</b> <em className={d.seed_med === 0 ? "flat" : d.seed_med < 0 ? "up" : "down"}>{d.seed_med === 0 ? "≈" : `${d.seed_med < 0 ? "↑" : "↓"}${Math.abs(d.seed_med)}`}</em></span>) : <span className="dim">–</span>;
    return <div className={`orow ${cls}`}><span className="who">{label}</span>{cell("seed1")}{cell("top2")}{cell("playoffs")}{sm}</div>;
  };

  return (
    <div className="trades">
      <header className="tr-hd">
        <h2>Trades</h2>
        <p>Pick a partner, then click players on both sides. Gains are rest-of-season starting-lineup points (regular season and playoffs), calibrated to how past predictions played out.</p>
      </header>
      {!engineUp && <p className="tr-off">Start the engine to evaluate trades.</p>}
      <Partners {...p} fits={fits} />
      {partner == null ? <><CallSheet {...p} /><Disagree {...p} /></> : (
        <div className="desk">
          <Roster rows={myRows} picked={deal.give} onToggle={toggle("give")} title="Your roster" side="me" colors={p.teamColors} flags={flags} />
          <section className="center">
            <div className="tray">
              <div className="side me"><span className="lab">You send</span>
                {deal.give.length ? deal.give.map((id) => <span key={id} className="pill">{byId.get(id)?.name}<button type="button" aria-label="Remove" onClick={() => toggle("give")(id)}>×</button></span>) : <span className="ph">Click your players</span>}</div>
              <div className="swap" aria-hidden="true">⇄</div>
              <div className="side them"><span className="lab">{pname} sends</span>
                {deal.get.length ? deal.get.map((id) => <span key={id} className="pill">{byId.get(id)?.name}<button type="button" aria-label="Remove" onClick={() => toggle("get")(id)}>×</button></span>) : <span className="ph">Click their players</span>}</div>
            </div>
            {err && <p className="tr-off">{err}</p>}
            {!ev && deal.give.length > 0 && deal.get.length > 0 && !err && <p className="dim">Evaluating…</p>}
            {ev && m && mt && (
              <>
                <GainChart ev={ev} me={me} partner={partner} pname={pname} />
                {verdict()}
                <div className="facts">
                  <div><span className="k">Right this often</span><b className="num">{pct(m.p_direction_right)}</b><span className={`conf ${m.edge_label === "noise" ? "noise" : m.edge_label === "thin edge" ? "thin" : "yes"}`}>{m.edge_label}</span></div>
                  <div><span className="k">Season · playoffs</span><b className="num">{signed(m.delta_regular_season)} · {signed(m.delta_playoffs)}</b><span className="dim">raw points</span></div>
                  <div><span className="k">Your offers to them</span><b className={`h ${h.tone}`}>{h.text}</b><span className="dim">2021–25</span></div>
                </div>
                {(m.would_drop.length > 0 || mt.would_drop.length > 0) && <p className="dim drops">Roster limits: {m.would_drop.length ? `you'd drop ${m.would_drop.join(", ")}` : ""}{m.would_drop.length && mt.would_drop.length ? "; " : ""}{mt.would_drop.length ? `${pname} would drop ${mt.would_drop.join(", ")}` : ""}.</p>}
                <div className="odds">
                  <div className="orow head"><span className="who">After trade</span><span>#1 seed</span><span>Top-2</span><span>Playoffs</span><span>Median seed</span></div>
                  {oddsRow(me, "You", "me")}
                  {oddsRow(partner, pname, "them")}
                  <p className="dim">{oddsBusy ? "Simulating 2,000 seasons… (about 10 seconds the first time)" : odds ? `Same simulated seasons as the League screen, so changes come from the trade. ≈ means inside noise.${odds.week != null && odds.week < 6 ? " Early season: seed and playoff odds only gain real skill around week 6 in backtests, so read these loosely." : ""}` : ""}</p>
                </div>
                <div className="ins-acts">
                  <button className="tbtn" type="button" onClick={() => p.onCopy?.(markdown())}>Copy for Claude</button>
                  <button className="tbtn" type="button" onClick={() => setDeal({ ...deal, give: [], get: [] })}>Clear deal</button>
                  {p.onLog && <button className="tbtn" type="button" onClick={() => p.onLog!({ kind: "trade", week: p.week,
                    title: `Give ${m.gives.join(", ")} for ${m.gets.join(", ")} (${pname})`,
                    payload: { partner, give_ids: deal.give, get_ids: deal.get },
                    model: { gain_cal: m.delta_total_calibrated, espn_partner: et?.delta_total },
                    model_p: m.delta_total_calibrated >= 0 ? m.p_direction_right : 1 - m.p_direction_right,
                    context: `Model: ${signed(m.delta_total_calibrated)} for you (${m.edge_label}). ESPN's view for ${pname}: ${signed(et?.delta_total)}. History: ${h.text}.` })}>Log to journal…</button>}
                </div>
                <div className="why">
                  {Object.entries(m.why).map(([n, w]) => <div key={n}><b>{n}</b><span>{w}</span></div>)}
                </div>
                <DealSoFar an={p.an} give={deal.give} get={deal.get} pname={pname} />
              </>
            )}
          </section>
          <Roster rows={theirRows} picked={deal.get} onToggle={toggle("get")} title={`${pname}`} side="them" colors={p.teamColors} flags={flags} />
        </div>
      )}
    </div>
  );
}

/* empty desk: a call sheet. Every partner, ordered by how well your rosters complement each
   other times how they've answered your past offers; the chip is their realistic spare part. */
function CallSheet({ me, teams, live, history, setDeal }: Props) {
  const a = live?.analysis;
  if (!a || !live) return <p className="tr-empty">Choose a team above, or open a trade fit from the League screen. {live ? "" : "(Loading league rosters for the call sheet…)"}</p>;
  const pos = ["QB", "RB", "WR", "TE"];
  const ids = Object.keys(a.pos_ppw).map(Number);
  const mean = Object.fromEntries(pos.map((p) => [p, ids.reduce((s, t) => s + a.pos_ppw[t][p], 0) / ids.length]));
  const dev = (t: number, p: string) => a.pos_ppw[t][p] - mean[p];
  const weight = (h?: TradeHistory) => (!h || !h.sent ? 1 : h.accepted ? 1.2 : h.declined ? 0.6 : 0.35);
  const rows = ids.filter((t) => t !== me).map((t) => {
    const get = pos.map((p) => ({ p, v: Math.min(-dev(me, p), dev(t, p)), mine: dev(me, p), theirs: dev(t, p) })).filter((x) => x.v > 0).sort((x, y) => y.v - x.v);
    const give = pos.map((p) => ({ p, v: Math.min(dev(me, p), -dev(t, p)), mine: dev(me, p), theirs: dev(t, p) })).filter((x) => x.v > 0).sort((x, y) => y.v - x.v);
    const fit = get.reduce((s, x) => s + x.v, 0) + give.reduce((s, x) => s + x.v, 0);
    const both = get.length > 0 && give.length > 0;
    const chip = get[0] ? [...(live.rosters[t] ?? [])].filter((r) => r.pos === get[0].p && !r.ir).sort((x, y) => (y.est_ppg ?? 0) - (x.est_ppg ?? 0))[1] : undefined;
    return { t, get, give, fit, both, score: (both ? fit : fit * 0.4) * weight(history[t]), chip, odds: live.odds[t] };
  }).sort((x, y) => y.score - x.score);
  const mx = Math.max(1, ...rows.map((r) => r.score));
  const name = (t: number) => teams.find((x) => x.id === t)?.name ?? `Team ${t}`;
  return (
    <section className="calls" aria-label="Who to call">
      <h3>Who to call <span>ordered by roster fit × how they've answered your offers (2021–25)</span></h3>
      <div className="call-hd"><span>Team</span><span>They can spare → you need</span><span>You can spare → they need</span><span>Their spare part</span><span>History</span><span>Playoffs</span><span /></div>
      {rows.map((r) => {
        const h = historyLabel(history[r.t]);
        return (
          <button key={r.t} type="button" className={`call ${r.both ? "" : "half"}`} onClick={() => setDeal({ partner: r.t, give: [], get: r.chip ? [r.chip.id] : [] })}>
            <span className="cn"><i className="cbar" style={{ width: `${(r.score / mx) * 100}%` }} /><b>{name(r.t)}</b></span>
            <span className="pz">{r.get.length ? r.get.slice(0, 2).map((x) => <i key={x.p} title={`${x.p}: they're ${signed(x.theirs)} per week vs the league, you're ${signed(x.mine)}`} style={{ ["--c" as string]: POS_VAR[x.p] }}>{x.p} <em>{signed(x.theirs)}</em></i>) : <span className="dim">nothing you need</span>}</span>
            <span className="pz">{r.give.length ? r.give.slice(0, 2).map((x) => <i key={x.p} title={`${x.p}: you're ${signed(x.mine)} per week vs the league, they're ${signed(x.theirs)}`} style={{ ["--c" as string]: POS_VAR[x.p] }}>{x.p} <em>{signed(x.theirs)}</em></i>) : <span className="dim">nothing they need</span>}</span>
            <span className="chip">{r.chip ? <>{r.chip.name} <small className="dim">{f1(r.chip.est_ppg)} ppg</small></> : <span className="dim">–</span>}</span>
            <span className={`h ${h.tone}`}>{h.text}</span>
            <span className="num">{pct(r.odds?.playoffs)}</span>
            <span className="go">Open desk →</span>
          </button>
        );
      })}
      <p className="tr-empty">Numbers are starter points per week against the league average by position, from the season simulation. "Spare part" = their second-best healthy player where they're deep, the kind of player teams part with. Opening a desk starts the deal with him on their side.</p>
    </section>
  );
}

export function fitTeams(a: LeagueAnalysis | undefined, me: number): Set<number> {
  if (!a) return new Set();
  const pos = ["QB", "RB", "WR", "TE"];
  const ids = Object.keys(a.pos_ppw).map(Number);
  const mean = Object.fromEntries(pos.map((p) => [p, ids.reduce((s, t) => s + a.pos_ppw[t][p], 0) / ids.length]));
  const dev = (t: number, p: string) => a.pos_ppw[t][p] - mean[p];
  return new Set(ids.filter((t) => t !== me && pos.some((p) => dev(me, p) < 0 && dev(t, p) > 0) && pos.some((p) => dev(me, p) > 0 && dev(t, p) < 0))
    .map((t) => ({ t, s: pos.reduce((s, p) => s + Math.max(0, Math.min(-dev(me, p), dev(t, p))) + Math.max(0, Math.min(dev(me, p), -dev(t, p))), 0) }))
    .sort((x, y) => y.s - x.s).slice(0, 3).map((x) => x.t));
}

/* inspector: idea search */
export function IdeasPanel({ partner, pname, engineUp, onLoad }: { partner: number | null; pname: string; engineUp: boolean; onLoad: (partner: number, give: number[], get: number[]) => void }) {
  const [st, setSt] = useState<IdeasState | null>(null);
  useEffect(() => { if (engineUp) tradeApi.ideas().then(setSt, () => undefined); }, [engineUp]);
  useEffect(() => {
    if (st?.state !== "running") return;
    const t = window.setInterval(() => tradeApi.ideas().then(setSt, () => undefined), 2000);
    return () => window.clearInterval(t);
  }, [st?.state]);
  const start = async (p: number | null) => setSt(await tradeApi.startIdeas(p));
  const secs = st?.state === "running" && st.started ? Math.round(Date.now() / 1000 - st.started) : 0;
  return (
    <div className="ideas">
      <div><div style={{ font: "700 17px/1.2 var(--sys)" }}>Trade ideas</div>
        <p className="ins-note">Searches 1-for-1, 2-for-1 and 1-for-2 deals where both sides gain by our numbers, ranked by your calibrated gain.</p></div>
      <div className="ins-acts">
        <button className="tbtn pri" type="button" disabled={!engineUp || partner == null || st?.state === "running"} onClick={() => start(partner)}>With {pname || "partner"}</button>
        <button className="tbtn" type="button" disabled={!engineUp || st?.state === "running"} onClick={() => start(null)}>Whole league (~1 min)</button>
      </div>
      {st?.state === "running" && <p className="ins-note">Searching… {secs}s</p>}
      {st?.state === "failed" && <p className="ins-note">Search failed: {st.error}</p>}
      {st?.state === "done" && (st.rows ?? []).length === 0 && <p className="ins-note">No deal helps both sides by our numbers.</p>}
      <div className="ilist">
        {(st?.rows ?? []).map((r, i) => (
          <button key={i} type="button" className="idea" title={r.why} onClick={() => onLoad(r.partner, r.give_ids, r.get_ids)} disabled={r.give_ids.some((x) => x == null) || r.get_ids.some((x) => x == null)}>
            <span className="w">{r.with}</span>
            <span className="g"><b>{r.give}</b> → <b>{r.get}</b></span>
            <span className="n"><span className="num you">{signed(r.your_gain_calibrated)}</span> you · <span className="num them">{signed(r.their_gain)}</span> them · {pct(r.p_right)} · <span className={`conf ${r.trust === "noise" ? "noise" : r.trust === "thin edge" ? "thin" : "yes"}`}>{r.trust}</span></span>
            {r.you_drop && <span className="d">you'd drop {r.you_drop}</span>}
          </button>
        ))}
      </div>
    </div>
  );
}

/* The deal's players, season so far: what each one's app shows (points per game), what his usage
   usually produces, and the model's rest-of-season number. Scoring over usage fades (~9% carries),
   and the model already discounts it, so the gap is a read on perception, not an edge. */
function DealSoFar({ an, give, get, pname }: { an?: Analytics | null; give: number[]; get: number[]; pname: string }) {
  if (!an) return null;
  const by = new Map(an.players.map((x) => [x.id, x]));
  const rows: { side: string; p: APlayer }[] = [...give.map((id) => ({ side: "You send", id })), ...get.map((id) => ({ side: `${pname} sends`, id }))]
    .map(({ side, id }) => ({ side, p: by.get(id) })).filter((x): x is { side: string; p: APlayer } => !!x.p);
  if (!rows.length) return null;
  return (
    <div className="dsf">
      <h5>Season so far <span>what their app shows vs what holds up</span></h5>
      <table>
        <thead><tr><th /><th>So far</th><th>Usage <EvFor m={an.metrics} k="fpoe" /></th><th>Model</th></tr></thead>
        <tbody>{rows.map(({ side, p }) => {
          const fl = rowFlag(p.flags);
          return <tr key={p.id}><th><b>{p.name}</b><small>{side}</small>{fl && <FlagChip f={fl} />}</th><td className="num">{f1(p.ppg)}</td><td className="num">{f1(p.xfp)}</td><td className="num"><b>{f1(p.est_ppg)}</b></td></tr>;
        })}</tbody>
      </table>
      <p className="dim">Points per game is what a partner sees first. Scoring over usage has carried forward only about {Math.round(100 * (an.metrics.fpoe?.sticks ?? 0.09))}% in 2021–25, and the model's number (used for the gains above) already discounts it. A hot player looks better on their app than on the model: useful when you're sending him, a warning when you're getting him.</p>
    </div>
  );
}

/* Empty desk, below the call sheet: where the season so far and the model disagree most. Not a
   model edge (the model's number is the one that held up); a list of players another manager may
   read differently from us. */
function Disagree({ me, teams, an, setDeal }: Props) {
  if (!an) return null;
  const name = (t: number | null) => teams.find((x) => x.id === t)?.name ?? "free agent";
  const pool = an.players.filter((x) => x.owner != null && x.games >= 3 && x.ppg != null && x.est_ppg != null && x.pool);
  const gap = (x: APlayer) => (x.ppg ?? 0) - (x.est_ppg ?? 0);
  const sell = pool.filter((x) => x.owner === me && gap(x) >= 2.5).sort((a, b) => gap(b) - gap(a)).slice(0, 6);
  const buy = pool.filter((x) => x.owner !== me && gap(x) <= -2.5).sort((a, b) => gap(a) - gap(b)).slice(0, 8);
  const row = (x: APlayer, mine: boolean) => (
    <button key={x.id} type="button" className="dg-row" disabled={mine} onClick={() => !mine && setDeal({ partner: x.owner, give: [], get: [x.id] })}
      title={`${x.name}: ${f1(x.ppg)} ppg so far, model ${f1(x.est_ppg)} rest of season, usage ${f1(x.xfp)}`}>
      <span className="nm"><b>{x.name}</b> <small>{x.pos} · {mine ? "yours" : name(x.owner)}</small></span>
      <span className="num">{f1(x.ppg)}</span><span className="num">{f1(x.xfp)}</span><span className="num"><b>{f1(x.est_ppg)}</b></span>
      <span className={`num gap ${gap(x) > 0 ? "up" : "down"}`}>{signed(gap(x))}</span>
      {!mine && <span className="go">Open desk →</span>}
    </button>
  );
  const hd = <div className="dg-row hd"><span /><span>So far</span><span>Usage</span><span>Model</span><span>Gap</span></div>;
  return (
    <section className="calls dg" aria-label="Where the season and the model disagree">
      <h3>Where the season and the model disagree <Ev l="descriptive" tip="Not a model edge: the model's rest-of-season number is what held up in backtests. This lists where points per game so far and the model differ most." /><span>points per game so far vs the model's rest of season, rostered players with 3+ games</span></h3>
      <div className="dg-two">
        <div><h5>Yours, scoring above the model</h5>{hd}{sell.length ? sell.map((x) => row(x, true)) : <p className="dim">None of your players are 2.5+ ppg above the model.</p>}</div>
        <div><h5>Theirs, scoring below the model</h5>{hd}{buy.length ? buy.map((x) => row(x, false)) : <p className="dim">Nobody is 2.5+ ppg below the model.</p>}</div>
      </div>
      <p className="tr-empty">A player whose points per game run ahead of the model looks better on a partner's app than on ours, and the reverse. Gaps like these mostly come from scoring over usage, which fades (about {Math.round(100 * (an.metrics.fpoe?.sticks ?? 0.09))}% carries), from injuries the model expects to heal, or from role changes it has already priced. Check the reason in the player file before you act.</p>
    </section>
  );
}

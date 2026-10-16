import { useFile } from "../../components/FileContext";
import type { LeagueLive, OutItem, PowerTeam, RosterRow, WhatIfResult } from "../../types";
import { f1, pct, signed } from "../../format";
import { Face } from "../../components/bits";
import "./whatif.css";

const OPTS: { label: string; weeks: number | null | undefined }[] = [
  { label: "In", weeks: undefined }, { label: "1 wk", weeks: 1 }, { label: "2 wk", weeks: 2 }, { label: "4 wk", weeks: 4 }, { label: "Season", weeks: null },
];

interface Props {
  team: PowerTeam;
  isMe: boolean;
  isOpp: boolean;
  live: LeagueLive | null;
  engineUp: boolean;
  whatif: WhatIfResult | null;
  scenario: OutItem[];
  busy: boolean;
  onSet: (p: RosterRow, weeks: number | null | undefined) => void;
}

/** What-if console (job: test). Readouts for the selected team, then a switch per player. */
export function WhatIfPanel({ team, isMe, isOpp, live, engineUp, whatif, scenario, busy, onSet }: Props) {
  const base = live?.odds[team.id], now = whatif?.scenario[team.id] ?? base, d = whatif?.delta[team.id];
  const roster = live?.rosters[team.id] ?? [];
  const { open: openFile } = useFile();
  const noise = (whatif?.noise_pp ?? live?.noise_pp ?? 1.1) / 100;
  const outOf = (id: number) => scenario.find((o) => o.id === id);
  const readout = (label: string, k: "top2" | "playoffs" | "title" | "seed1") => (
    <div className="ro">
      <span className="k">{label}</span>
      <span className="v">{now ? pct(now[k]) : "–"}</span>
      <span className={`dd ${d && Math.abs(d[k]) >= 2 * noise ? (d[k] > 0 ? "up" : "down") : ""}`}>
        {d ? (Math.abs(d[k]) < 2 * noise ? "≈ no change" : `${d[k] > 0 ? "+" : "−"}${Math.round(Math.abs(d[k]) * 100)} pts`) : " "}
      </span>
    </div>
  );
  return (
    <div className="wi">
      <div><div className="wi-name">{team.name}</div>
        <div className="dim">{isMe ? "Your team" : isOpp ? "This week's opponent" : "League"} · {team.record.w}–{team.record.l} · {f1(team.pf)} pts</div></div>
      <div className={`lcd ${busy ? "busy" : ""}`} aria-live="polite">
        {readout("#1 seed", "seed1")}{readout("Top-2", "top2")}{readout("Playoffs", "playoffs")}{readout("Title", "title")}
        <div className="foot">{now ? `Proj. final ${f1(now.reg_pts)}${d && Math.abs(d.reg_pts) >= 1 ? ` (${signed(d.reg_pts)})` : ""}` : engineUp ? "Simulating…" : "Engine not running"}</div>
      </div>
      <section>
        <h6>What if he misses… <span>weeks, from this week</span></h6>
        {!engineUp && <p className="ins-note">Start the engine to run what-ifs.</p>}
        <div className="wi-list">
          {roster.filter((r) => !r.ir).map((r) => {
            const o = outOf(r.id);
            return (
              <div key={r.id} className={`wi-row ${o ? "out" : ""}`}>
                <Face p={{ ...r, headshot: r.headshot }} className="round" />
                <div className="who"><button type="button" className="wi-name" onClick={() => openFile(r.id, roster.filter((x) => !x.ir).map((x) => x.id))} title="Open player file">{r.name}</button><span>{r.pos} · {f1(r.est_ppg)} ppg · plays {pct(r.avail)}</span></div>
                <div className="sw" role="group" aria-label={`${r.name} games missed`}>
                  {OPTS.map((op) => {
                    const on = op.weeks === undefined ? !o : !!o && o.weeks === op.weeks;
                    return <button key={op.label} type="button" aria-pressed={on} disabled={!engineUp || busy} onClick={() => onSet(r, op.weeks)}>{op.label}</button>;
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </section>
      <p className="ins-note">The replacement is the best healthy player on his roster, or a waiver-level streamer. Scenarios stack across teams. Every scenario reuses the same simulated seasons, so differences come from the scenario itself.</p>
    </div>
  );
}

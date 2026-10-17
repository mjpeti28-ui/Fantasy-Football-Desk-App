import { ord } from "./Card";
import { Sec, Tile, n2 } from "./parts";
import type { MxPffDef, MxPffDefense, MxPffLine, MxPffPlayer, MxPffUnit } from "./types";

/* PFF matchup context: who a player lines up against, and how the units compare. Descriptive only.
   backtest/matchup_pff.py (2023–25) found none of it moves a player beyond ESPN's number, so nothing here
   feeds the card's projection. PFF has no left/right side or shadow flag for corners. */

const BACKTEST = "Context, not in the number: in 2023–25 backtests none of these matchups (coverage faced by alignment, pass rush vs pass blocking, run defense vs run blocking) predicted anything ESPN’s projection had missed.";
const g1 = (x: number | null | undefined) => (x == null ? "–" : x.toFixed(1));

function Unit({ k, u, sub }: { k: string; u?: MxPffUnit; sub?: string }) {
  return <Tile k={k} v={u ? g1(u.grade) : "–"} sub={u ? `${ord(u.rank)} of ${u.of}${sub ? ` · ${sub}` : ""}` : sub} />;
}

function Cover({ rows, title, hot }: { rows: MxPffDef[]; title: string; hot?: boolean }) {
  if (!rows.length) return null;
  return (
    <>
      <h6>{title}{hot && <span className="mx-pff-hot"> · where he lines up most</span>}</h6>
      <table className={`mx-tbl ${hot ? "hot" : ""}`}><thead><tr><th /><th>pos</th><th title="Share of the team’s snaps at this spot">share</th><th title="PFF coverage grade">cov</th><th>tgt</th><th title="Yards allowed per coverage snap">yds/cs</th><th>td</th><th>int</th></tr></thead><tbody>
        {rows.map((x) => <tr key={x.pff_id}><td>{x.name}</td><td className="dim">{x.pos}</td><td className="num dim">{x.share == null ? "–" : `${x.share}%`}</td><td className="num">{g1(x.cov_grade)}</td><td className="num">{x.tgt}</td><td className="num">{n2(x.yds_cov)}</td><td className="num">{x.td}</td><td className="num">{x.int}</td></tr>)}
      </tbody></table>
    </>
  );
}

function Rushers({ rows }: { rows: MxPffDef[] }) {
  return (
    <table className="mx-tbl"><thead><tr><th /><th>pos</th><th title="PFF pass-rush grade">rush</th><th>press</th><th title="Pressures per pass-rush snap">rate</th><th>sk</th></tr></thead><tbody>
      {rows.map((x) => <tr key={x.pff_id}><td>{x.name}</td><td className="dim">{x.pos}</td><td className="num">{g1(x.rush_grade)}</td><td className="num">{x.pressures}</td><td className="num">{x.prp == null ? "–" : `${x.prp.toFixed(0)}%`}</td><td className="num">{x.sacks ?? 0}</td></tr>)}
    </tbody></table>
  );
}

function RunD({ rows }: { rows: MxPffDef[] }) {
  return (
    <table className="mx-tbl"><thead><tr><th /><th>pos</th><th title="PFF run-defense grade">run</th><th title="Tackles that count as an offensive failure">stops</th><th>snaps</th></tr></thead><tbody>
      {rows.map((x) => <tr key={x.pff_id}><td>{x.name}</td><td className="dim">{x.pos}</td><td className="num">{g1(x.run_grade)}</td><td className="num">{x.stops}</td><td className="num dim">{x.run_snaps}</td></tr>)}
    </tbody></table>
  );
}

function Line({ rows, kind }: { rows: MxPffLine[]; kind: "pass" | "run" }) {
  if (!rows.length) return <p className="dim">No line data.</p>;
  return (
    <table className="mx-tbl"><thead><tr><th /><th>at</th><th>{kind === "pass" ? "pass blk" : "run blk"}</th>{kind === "pass" && <><th>press</th><th>sk</th></>}</tr></thead><tbody>
      {rows.map((x) => <tr key={`${x.name}-${x.at}`}><td>{x.name}</td><td className="dim">{x.at}</td><td className="num">{g1(kind === "pass" ? x.pass_grade : x.run_grade)}</td>{kind === "pass" && <><td className="num">{x.pressures}</td><td className="num">{x.sacks}</td></>}</tr>)}
    </tbody></table>
  );
}

/** "Who he'll see": the player's side of the PFF matchup, under the start/sit card. */
export function PffSee({ x, name }: { x: MxPffPlayer; name: string }) {
  const first = name.split(" ")[0];
  const aside = `PFF grades · ${x.season} through week ${x.through ?? "?"} · 1st = best unit · context`;
  const units = (
    <div className="mx-tiles">
      {(x.pos === "WR" || x.pos === "TE" || x.pos === "QB") && <Unit k={`${x.opp} coverage`} u={x.them.coverage} />}
      {x.pos !== "RB" && <Unit k={`${x.opp} pass rush`} u={x.them.pass_rush} />}
      {x.pos !== "RB" && <Unit k="His line, pass blocking" u={x.us.pass_block} />}
      {x.pos === "RB" && <Unit k={`${x.opp} run defense`} u={x.them.run_defense} />}
      {x.pos === "RB" && <Unit k="His line, run blocking" u={x.us.run_block} />}
      {x.pos === "RB" && <Unit k={`${x.opp} pass rush`} u={x.them.pass_rush} />}
    </div>
  );
  return (
    <section className="mx-pff" aria-label="Who he will see">
      <Sec title="Who he’ll see" aside={aside}>
        {units}
        {(x.pos === "WR" || x.pos === "TE") && (
          <div className="mx-cols">
            <div>
              {x.align
                ? <p className="mx-note t">{first} lines up in the slot on <b>{x.align.slot}%</b> of pass snaps, out wide on <b>{x.align.wide}%</b>{x.align.inline >= 10 ? <>, inline on <b>{x.align.inline}%</b></> : null}.
                  {x.likely === "slot" ? ` So ${x.opp}’s slot defenders are the ones he sees most.` : x.likely === "outside" ? ` So he mostly sees ${x.opp}’s outside corners; PFF doesn’t say which side, so it could be either.` : " Inline, he is mostly covered by linebackers and safeties."}</p>
                : <p className="mx-note t">No PFF alignment for {first} yet.</p>}
              <Cover rows={x.outside ?? []} title={`${x.opp} outside corners`} hot={x.likely === "outside"} />
              <Cover rows={x.slot ?? []} title={`${x.opp} slot defenders`} hot={x.likely === "slot"} />
            </div>
            <div>
              <h6>Who covered him this season</h6>
              {x.covered_by && x.covered_by.length
                ? <table className="mx-tbl"><thead><tr><th /><th>team</th><th>tgt</th><th>rec</th><th>yds</th><th>td</th></tr></thead><tbody>
                    {x.covered_by.map((c, i) => <tr key={i} className={c.theirs ? "hot" : ""}><td>{c.name ?? "?"}</td><td className="dim">{c.team ?? ""}</td><td className="num">{c.tgt}</td><td className="num">{c.rec}</td><td className="num">{c.yds}</td><td className="num">{c.td}</td></tr>)}
                  </tbody></table>
                : <p className="dim">No targets charted against a defender yet.</p>}
              <p className="mx-note r">Targets PFF charted with each defender in coverage. Small samples: a few targets each.</p>
            </div>
          </div>
        )}
        {x.pos === "QB" && (
          <div className="mx-cols">
            <div><h6>{x.opp} pass rushers</h6><Rushers rows={x.rushers ?? []} /></div>
            <div><h6>His line</h6><Line rows={x.line} kind="pass" /></div>
          </div>
        )}
        {x.pos === "RB" && (
          <div className="mx-cols">
            <div><h6>{x.opp} run defenders</h6><RunD rows={x.run ?? []} /></div>
            <div><h6>His line</h6><Line rows={x.line} kind="run" /></div>
          </div>
        )}
        <p className="mx-note r">{BACKTEST} PFF has no left/right side or shadow flag for corners.</p>
      </Sec>
    </section>
  );
}

/** PFF tab in a defense's "Their defenders". */
export function PffDefenders({ p }: { p: MxPffDefense }) {
  return (
    <>
      <p className="mx-note t">PFF grades, {p.season} through week {p.through ?? "?"}. {BACKTEST}</p>
      <div className="mx-tiles">
        <Unit k="Coverage" u={p.units.coverage} /><Unit k="Pass rush" u={p.units.pass_rush} /><Unit k="Run defense" u={p.units.run_defense} />
      </div>
      <Cover rows={p.outside} title="Outside corners" />
      <Cover rows={p.slot} title="Slot defenders" />
      {p.rushers.length > 0 && <><h6>Pass rushers</h6><Rushers rows={p.rushers} /></>}
      {p.run.length > 0 && <><h6>Run defenders</h6><RunD rows={p.run} /></>}
    </>
  );
}

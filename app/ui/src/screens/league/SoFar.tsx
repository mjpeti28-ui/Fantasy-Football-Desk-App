import { useMemo } from "react";
import type { ALeague, ALeagueTeam } from "../../types";
import { f1, signed } from "../../format";
import { Ev } from "../../components/Analytics";

/* League › Season so far: what has happened, team by team and week by week, from the same tables as
   My Team › Analysis. All of it is descriptive: lineup efficiency (r .11), luck (r −.13) and weekly
   swing (r −.05) did not repeat in 2021–25, and the chips say so. Same almanac cards as the analysis. */

interface Props { lg: ALeague; myId: number; oppId: number | null; onSelect: (id: number) => void }
const role = (id: number, myId: number, oppId: number | null) => (id === myId ? "me" : id === oppId ? "opp" : "");
const r2 = (x: number | null | undefined) => (x == null ? "–" : x.toFixed(2).replace("-", "−"));

function Heat({ lg, myId, oppId, onSelect }: Props) {
  const weeks = Array.from({ length: lg.through_week }, (_, i) => i + 1);
  const cell = useMemo(() => new Map(lg.cells.map((c) => [`${c.team_id}-${c.week}`, c])), [lg]);
  const order = [...lg.teams].sort((a, b) => b.pts - a.pts);
  const sw = lg.labels.sd;
  return (
    <section className="an-card wide sf-heat">
      <header><h4>Every team, every week <Ev l="descriptive" tip="What happened. Weekly swing didn't repeat in 2021–25." /></h4>
        <p>Points scored; green above that week's league average, red below, deeper = further (in standard deviations). The number is the score; a bar along the top marks that week's high score.
          Week-to-week swing hasn't carried forward (r {r2(sw?.repeat)}), so a hot or cold run here says little about the next one.</p></header>
      <div className="mx-wrap">
        <table className="mx">
          <thead><tr><th />{weeks.map((w) => <th key={w}>Wk {w}</th>)}<th>Total</th></tr></thead>
          <tbody>
            {order.map((t) => (
              <tr key={t.id} className={role(t.id, myId, oppId)} onClick={() => onSelect(t.id)}>
                <th className="tn">{t.name}</th>
                {weeks.map((w) => {
                  const c = cell.get(`${t.id}-${w}`);
                  if (!c) return <td key={w} className="tot">–</td>;
                  const a = Math.min(1, Math.abs(c.z) / 2.2);
                  return <td key={w} className={`${c.z >= 0 ? "pos" : "neg"} ${c.rank === 1 ? "top" : ""}`} style={{ ["--a" as string]: a * 0.85 }}
                    title={`${t.name}, week ${w}: ${f1(c.pts)} (#${c.rank} of ${lg.n_teams}, ${signed(c.z)} sd vs the league that week); best possible ${f1(c.opt)}`}>{Math.round(c.pts)}</td>;
                })}
                <td className="tot">{f1(t.pts)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Bench({ lg, myId, oppId, onSelect }: Props) {
  const order = [...lg.teams].sort((a, b) => b.opt - a.opt);
  const mx = Math.max(...order.map((t) => t.opt));
  const eff = lg.labels.eff;
  const worst = [...lg.teams].sort((a, b) => b.left - a.left)[0];
  return (
    <section className="an-card">
      <header><h4>Scored vs best possible <Ev l="noise" tip={`Lineup efficiency repeated at r ${r2(eff?.repeat)} in 2021–25: not a stable habit.`} /></h4>
        <p>Solid = points your lineups scored; hatched = what the best lineup from the same roster would have added. Seeding is by total points, so every one counts.
          Leaving points hasn't been a repeatable habit (r {r2(eff?.repeat)}).</p></header>
      <div className="sf-bars">
        {order.map((t) => (
          <button key={t.id} type="button" className={`sf-bar ${role(t.id, myId, oppId)}`} onClick={() => onSelect(t.id)}
            title={`${t.name}: ${f1(t.pts)} scored of ${f1(t.opt)} possible (${Math.round(100 * t.pts / t.opt)}%), ${f1(t.left)} left on the bench`}>
            <span className="tn">{t.name}</span>
            <span className="trk"><i className="got" style={{ width: `${(100 * t.pts) / mx}%` }} /><i className="left" style={{ width: `${(100 * t.left) / mx}%` }} /></span>
            <span className="num">{f1(t.left)}</span>
          </button>
        ))}
      </div>
      <p className="an-note">Number = points left on the bench. Most left: {worst.name} ({f1(worst.left)}). Bars ordered by best possible; ESPN's projection of each team's starters is in the main table's "vs projection".</p>
    </section>
  );
}

function Sources({ lg, myId, oppId, onSelect }: Props) {
  const order = [...lg.teams].sort((a, b) => b.drafted / b.pts - a.drafted / a.pts);
  return (
    <section className="an-card">
      <header><h4>Drafted vs added <Ev l="descriptive" tip="What happened." /></h4>
        <p>Share of each team's starting points from players it drafted (filled) vs players added later by waiver, free agency or trade (outline).</p></header>
      <div className="sf-bars">
        {order.map((t) => {
          const d = t.pts > 0 ? t.drafted / t.pts : 0;
          return (
            <button key={t.id} type="button" className={`sf-bar ${role(t.id, myId, oppId)}`} onClick={() => onSelect(t.id)}
              title={`${t.name}: ${Math.round(100 * d)}% of starting points from its own draft picks (${f1(t.drafted)} of ${f1(t.pts)})`}>
              <span className="tn">{t.name}</span>
              <span className="trk split"><i className="got" style={{ width: `${100 * d}%` }} /><i className="add" style={{ width: `${100 * (1 - d)}%` }} /></span>
              <span className="num">{Math.round(100 * (1 - d))}%</span>
            </button>
          );
        })}
      </div>
      <p className="an-note">Number = share from added players.</p>
    </section>
  );
}

function Draft({ lg, myId, oppId }: Omit<Props, "onSelect">) {
  const name = new Map(lg.teams.map((t: ALeagueTeam) => [t.id, t.name]));
  const scored = lg.draft.filter((p) => p.pos_rank_now != null && ["QB", "RB", "WR", "TE"].includes(p.pos));
  const hits = [...scored].sort((a, b) => (b.pos_pick - b.pos_rank_now!) - (a.pos_pick - a.pos_rank_now!)).slice(0, 8);
  const misses = [...scored].filter((p) => p.round <= 6).sort((a, b) => (a.pos_pick - a.pos_rank_now!) - (b.pos_pick - b.pos_rank_now!)).slice(0, 8);
  const row = (p: (typeof scored)[number]) => (
    <li key={p.id} className={role(p.team_id, myId, oppId)} title={`${p.name}: drafted ${p.pos}${p.pos_pick} (round ${p.round}, pick ${p.overall}) by ${name.get(p.team_id)}; ${p.pos}${p.pos_rank_now} in points so far (${f1(p.pts)})`}>
      <span className="nm">{p.name}</span><span className="dim">{name.get(p.team_id) ?? ""}</span>
      <span className="num">{p.pos}{p.pos_pick} → <b>{p.pos}{p.pos_rank_now}</b></span>
    </li>
  );
  return (
    <section className="an-card wide">
      <header><h4>Draft: where picks landed <Ev l="descriptive" tip="What happened. Ranks are total points so far, so missed games count against a player." /></h4>
        <p>Drafted position rank (the order each position came off the board) against the position rank in total points so far. Total points, so injuries and byes count.</p></header>
      <div className="sf-draft">
        <div><h5>Biggest risers</h5><ul>{hits.map(row)}</ul></div>
        <div><h5>Biggest fallers, rounds 1–6</h5><ul>{misses.map(row)}</ul></div>
      </div>
    </section>
  );
}

export function SoFar(p: Props) {
  return (
    <div className="analysis sf">
      <h3 className="an-h">Season so far <small>through week {p.lg.through_week} · what happened, labelled by whether it has repeated</small></h3>
      <Heat {...p} />
      <div className="an-row"><Bench {...p} /><Sources {...p} /></div>
      <Draft {...p} />
    </div>
  );
}

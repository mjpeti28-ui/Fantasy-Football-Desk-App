import type { Analytics, TeamWeek } from "../../types";
import type { MxBoard } from "./types";
import { lastName } from "../../format";
import { Ev } from "../../components/Analytics";

/* Matchups › Schedule ahead: your players' opponents from next week through 17, each cell graded by
   the defense's rating today (pass defense for QB/WR/TE, run defense for RBs). For planning holds and
   trades, not weekly start/sit (ESPN's weekly number already prices the matchup). Weak signal: a
   rating before week W predicted playoff pass defense at r .22 and run defense at .06 in 2023–25. */

const PO = [15, 16, 17];
type G = { label: string; rank: number; of: number } | null;

export function Ahead({ data, an, tw }: { data: MxBoard; an: Analytics | null; tw: TeamWeek }) {
  if (!an) return <p className="mx-empty">Season analytics are still building.</p>;
  const weeks = an.sched_weeks;
  const ev = an.sched_ev;
  const nfl = (t: string | null) => (t ? data.alias[t] ?? t : null);
  const grade = (opp: string, pos: string): G => {
    const t = data.teams[opp];
    const g = pos === "RB" ? t?.run_grade : t?.pass_grade;
    return g ? { label: g.label, rank: g.rank, of: g.of } : null;
  };
  const rows = tw.players.filter((p) => ["QB", "RB", "WR", "TE"].includes(p.pos) && p.slot !== "IR")
    .sort((a, b) => "QBRBWRTE".indexOf(a.pos) - "QBRBWRTE".indexOf(b.pos) || (b.est_ppg ?? 0) - (a.est_ppg ?? 0));
  const tip = `Ratings now vs what the defense allowed later, 2023–25: fantasy playoffs r ${ev.pass_playoffs?.v.toFixed(2) ?? "–"} (pass) and ${ev.run_playoffs?.v.toFixed(2) ?? "–"} (run); next four weeks ${ev.pass_next4?.v.toFixed(2) ?? "–"} / ${ev.run_next4?.v.toFixed(2) ?? "–"}. Fantasy points allowed by position did worse (${ev.fpa_playoffs?.v.toFixed(2) ?? "–"}).`;
  return (
    <section className="mx-ahead">
      <header>
        <h3>Schedule ahead <Ev l={ev.label} tip={tip} /> <span>your players' opponents, weeks {weeks[0]}–{weeks[weeks.length - 1]}; graded by each defense's rating today</span></h3>
        <p>Green = a soft defense for that position (pass defense for QB, WR, TE; run defense for RB), red = a tough one, rank of 32 printed. Weeks 15–17 are the playoffs.
          This is for planning holds and trades. It is a weak signal: ratings today predicted playoff pass defense at r {ev.pass_playoffs?.v.toFixed(2) ?? "–"} and run defense at {ev.run_playoffs?.v.toFixed(2) ?? "–"} in 2023–25,
          and ESPN's weekly projection already prices the matchup once the week arrives.</p>
      </header>
      <div className="mx-ah-wrap">
        <table className="mx-ah">
          <thead><tr><th />{weeks.map((w) => <th key={w} className={PO.includes(w) ? "po" : ""}>{w}</th>)}<th title="Soft / tough matchups in weeks 15–17">Playoffs</th></tr></thead>
          <tbody>
            {rows.map((p) => {
              const sc = an.schedule[nfl(p.team) ?? ""] ?? {};
              let soft = 0, tough = 0;
              return (
                <tr key={p.id}>
                  <th><b>{lastName(p.name)}</b> <small>{p.pos} · {p.team}</small></th>
                  {weeks.map((w) => {
                    const g = sc[String(w)];
                    if (!g) return <td key={w} className={`bye ${PO.includes(w) ? "po" : ""}`} title={`Week ${w}: bye`}>BYE</td>;
                    const gr = grade(g.opp, p.pos);
                    if (PO.includes(w) && gr) { if (gr.label === "Soft") soft++; if (gr.label === "Tough") tough++; }
                    return (
                      <td key={w} className={`${gr ? gr.label.toLowerCase() : ""} ${PO.includes(w) ? "po" : ""}`}
                        title={`Week ${w}: ${g.home ? "vs" : "@"} ${g.opp}${gr ? ` · ${p.pos === "RB" ? "run" : "pass"} defense #${gr.rank} of ${gr.of} (${gr.label.toLowerCase()}) today` : ""}`}>
                        <span className="op">{g.home ? "" : "@"}{g.opp}</span>{gr && <small>{gr.rank}</small>}
                      </td>
                    );
                  })}
                  <td className="sum">{soft ? <span className="s">{soft} soft</span> : null}{tough ? <span className="t">{tough} tough</span> : null}{!soft && !tough ? <span className="dim">–</span> : null}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function aheadMarkdown(data: MxBoard, an: Analytics | null, tw: TeamWeek): string {
  if (!an) return "";
  const nfl = (t: string | null) => (t ? data.alias[t] ?? t : null);
  const L = [`## Schedule ahead, weeks ${an.sched_weeks[0]}–17 (defense ratings today; weak signal: r ${an.sched_ev.pass_playoffs?.v.toFixed(2)} pass / ${an.sched_ev.run_playoffs?.v.toFixed(2)} run for the playoffs, 2023–25)`, "",
    `| Player | ${an.sched_weeks.join(" | ")} |`, `|---|${an.sched_weeks.map(() => "---").join("|")}|`];
  for (const p of tw.players.filter((x) => ["QB", "RB", "WR", "TE"].includes(x.pos) && x.slot !== "IR")) {
    const sc = an.schedule[nfl(p.team) ?? ""] ?? {};
    L.push(`| ${p.name} (${p.pos}) | ${an.sched_weeks.map((w) => { const g = sc[String(w)]; if (!g) return "bye"; const t = data.teams[g.opp]; const gr = p.pos === "RB" ? t?.run_grade : t?.pass_grade; return `${g.home ? "" : "@"}${g.opp}${gr ? ` ${gr.label[0]}${gr.rank}` : ""}`; }).join(" | ")} |`);
  }
  L.push("", "S/M/T = soft/middle/tough defense for that position today, with its rank of 32.");
  return L.join("\n");
}

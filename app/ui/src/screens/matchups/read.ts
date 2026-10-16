import type { MxBoard, MxCurrent, MxDefenseView, MxHist, MxPlayer, MxQbCurrent, MxQbHist } from "./types";
import { SHELLS, SHELL_LABEL } from "./types";
import { cardMarkdown } from "./Card";
import { lastOf, n2, p0, p1, pp, sg2 } from "./parts";

const tbl = (head: string[], rows: (string | number)[][]) => [`| ${head.join(" | ")} |`, `| ${head.map(() => "---").join(" | ")} |`, ...rows.map((r) => `| ${r.join(" | ")} |`)].join("\n");

export function mxMarkdown(board: MxBoard, p: MxPlayer | null, v: MxDefenseView | null, mode: "week" | "league", sel: string | null): string {
  const out: string[] = [];
  const attrib = `_Data: ${board.attribution}. Coverage splits are 2023–${lastOf(board.hist)}; the coverage file for ${board.season} is published after the season. Not PFF or PFR data._`;
  if (mode === "league" && !(sel && v)) {
    out.push(`# Defenses, week ${board.week} (${board.season})`, attrib, "");
    const rows = Object.entries(board.teams).sort((a, b) => (a[1].pass_grade?.rank ?? 99) - (b[1].pass_grade?.rank ?? 99));
    const gr = (g: { label: string; rank: number } | undefined) => (g ? `${g.label} (${g.rank})` : "–");
    out.push(tbl(["Defense", "Faces", "Vs the pass (1 = toughest)", "Vs the run", "Expected man", "Blitz (blend)", "Pressure (last yr)", "Games"],
      rows.map(([a, t]) => [a, board.schedule[a]?.opp ?? "bye", gr(t.pass_grade), gr(t.run_grade), p0(t.man), p0(t.blitz), p0(t.pressure), t.games])));
    out.push("", `Vs the pass / vs the run: one rating per defense, EPA allowed on every play this season so far blended with last season, ranked of 32 (1 = toughest; Tough is the top 11, Soft the bottom 11). ESPN’s projections already move with it.`,
      `League baseline for man coverage: ${p0(board.league.base.man)} (${board.league.base.season}). Expected man = baseline + the share of last season’s distance from the league that carries over.`);
    return out.join("\n");
  }
  if (p) {
    out.push(`# ${p.name} (${p.pos}, ${p.team ?? "–"}) vs ${p.opp ?? "bye"}, week ${board.week}`, attrib, "");
    if (p.card) out.push("## Start/sit card", ...cardMarkdown(p.card));
    if (p.pos === "QB") {
      const c = p.current as MxQbCurrent | null, h = p.hist as MxQbHist | null;
      if (c) out.push("", `## Passing this season`, `${c.db} dropbacks: EPA/dropback ${sg2(c.epa_db)}, success ${p0(c.success)}, completion ${p0(c.comp)}, ${n2(c.ypa, 1)} Y/A, aDOT ${n2(c.adot, 1)}, sack ${p1(c.sack)}, play-action ${p0(c.pa)}, vs blitz ${sg2(c.blitz_epa)} (${c.blitz_n}) vs no blitz ${sg2(c.noblitz_epa)}.`);
      if (h) out.push("", `## EPA per dropback by coverage (2023–25)`, tbl(["Split", "Dropbacks", "EPA/dropback", "± s.e.", "Comp", "Y/A", "Sack", "Pressure"],
        [["All", h.total], ["Man", h.man], ["Zone", h.zone], ...SHELLS.map((s) => [SHELL_LABEL[s], h.shells[s]] as const), ["Blitz", h.blitz], ["No blitz", h.no_blitz], ["Play-action", h.pa], ["No play-action", h.no_pa]]
          .map(([n, s]) => { const x = s as MxQbHist["total"]; return [n as string, x.db, sg2(x.epa_db), n2(x.se ?? null), p0(x.comp), n2(x.ypa, 1), p1(x.sack), p0(x.pressure)]; })));
    } else {
      const c = p.current as MxCurrent | null, h = p.hist as MxHist | null;
      if (c) out.push("", `## Usage this season`, `${c.tgt} targets in ${c.games} games (${p0(c.share)} of the team’s ${c.team_tgt}); ${c.rec} for ${c.yds} yards and ${c.td} TD; aDOT ${n2(c.adot, 1)}; EPA/target ${sg2(c.epa_t)}; catchable ${p0(c.catchable)}, contested ${c.contested}, drops ${c.drops}.`,
        `Target depth: behind LOS ${p0(c.depth.behind)}, 0–9 ${p0(c.depth.short)}, 10–19 ${p0(c.depth.mid)}, 20+ ${p0(c.depth.deep)}. Location: L ${p0(c.loc.left)} / M ${p0(c.loc.middle)} / R ${p0(c.loc.right)}. Share of his targets on play-action ${p0(c.pa_share)} (team ${p0(c.team_pa_share)}), screens ${p0(c.screen_share)} (team ${p0(c.team_screen_share)}), into a blitz ${p0(c.blitz_share)} (team ${p0(c.team_blitz_share)}).`);
      if (p.rush) out.push(`Rushing: ${p.rush.car} carries, ${n2(p.rush.ypc, 1)} per carry, EPA/rush ${sg2(p.rush.epa)}, success ${p0(p.rush.success)}, explosive ${p0(p.rush.explosive)}.`);
      if (h) {
        out.push("", `## Against coverage (2023–25; target rate = targets per dropback he is on the field for)`,
          tbl(["Coverage", "Dropbacks", "Targets", "Target rate", "PPR/target", "EPA/target"],
            [["All", h.total], ["Man", h.man], ["Zone", h.zone], ...SHELLS.map((s) => [SHELL_LABEL[s], h.shells[s]] as const), ["Blitz", h.other.blitz], ["No blitz", h.other.no_blitz], ["Play-action", h.other.pa], ["No play-action", h.other.no_pa]]
              .map(([n, s]) => { const x = s as MxHist["total"]; return [n as string, x.db, x.tgt, p1(x.tr), n2(x.ppr_t), sg2(x.epa_t)]; })),
          "", `Man minus zone target rate: ${pp(h.tr_diff)} raw, ${pp(h.tr_diff_shrunk)} after shrinking (signal ${p0(h.tr_rel)}). PPR/target man minus zone: ${sg2(h.ppr_diff)} raw, ${sg2(h.ppr_diff_shrunk)} shrunk (signal ${p0(h.ppr_rel)}): efficiency splits are mostly noise, usage splits partly real.`);
        if (h.routes.length) out.push("", tbl(["Route (targets)", "Share", "League share", "PPR/target"], h.routes.slice(0, 8).map((r) => [r.label, p0(r.share), p0(r.league_share), n2(r.ppr_t)])));
      }
    }
  }
  if (v) {
    const d = v.defense, c = d.current, lg = d.league;
    out.push("", `## ${d.abbr} defense`);
    const tg = board.teams[d.abbr];
    if (tg?.pass_grade && tg.run_grade) out.push(`Rating (1 = toughest of ${tg.pass_grade.of}): vs the pass ${tg.pass_grade.label} (rank ${tg.pass_grade.rank}); vs the run ${tg.run_grade.label} (rank ${tg.run_grade.rank}).`, "");
    out.push(tbl(["Season", "Dropbacks", "Man", "Cover 0/1", "Cover 2", "Cover 3", "Cover 4/6", "Blitz", "Pressure"],
      d.history.map((h) => [h.season, h.n, p0(h.man), p0(h.shell.C01), p0(h.shell.C2), p0(h.shell.C3), p0(h.shell.C46), p0(h.blitz), p0(h.pressure)])));
    if (d.expected) out.push(`Expected ${board.season}: man ${p0(d.expected.man)} (league ${p0(lg.base.man)}), ` + SHELLS.map((s) => `${SHELL_LABEL[s]} ${p0(d.expected!.shell[s])}`).join(", ") + ".");
    if (c) out.push(`This season (${c.games} games, ${c.n} dropbacks): EPA/dropback allowed ${sg2(c.epa_db)} (league ${sg2(lg.cur.epa_db)}), completion ${p0(c.comp)}, ${n2(c.ypa, 1)} Y/A, sack ${p1(c.sack)}, blitz ${p0(c.blitz)} (blend ${p0(c.blitz_exp)}).`);
    if (d.run) out.push(`Run defense: EPA/rush ${sg2(d.run.epa)} (league ${sg2(lg.cur_run.epa)}), ${n2(d.run.ypc, 1)} per carry, explosive ${p0(d.run.explosive)}.`);
    out.push(`Allowed by position (shrunk PPR/target vs league): ` + (["WR", "TE", "RB"] as const).map((k) => `${k} ${n2(d.allowed[k].ppr_t)} vs ${n2(d.allowed[k].league_ppr_t)} (${d.allowed[k].tgt} targets)`).join("; ") + ".");
    if (d.faced.length) out.push("", tbl(["Most PPR against them", "Pos", "Tgt", "Rec", "Yds", "TD", "PPR"], d.faced.map((f) => [f.name, f.pos, f.tgt, f.rec, f.yds, f.td, n2(f.ppr, 1)])));
    const pm = v.defenders.current.slice(0, 8);
    if (pm.length) out.push("", tbl(["Playmakers this season", "Pos", "Sacks", "Hits", "TFL", "INT", "PBU"], pm.map((x) => [x.name, x.pos, x.sacks, x.hit, x.tfl, x.int, x.pbu])));
  }
  return out.join("\n");
}

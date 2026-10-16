import type { ReactNode } from "react";
import { Status } from "../../components/bits";
import { f1 } from "../../format";
import { sg2 } from "./parts";
import type { MxCard, MxCardRow, MxGrade, MxTag } from "./types";

/* The start/sit card. One number leads (ESPN's), then how much of it is the matchup drawn against a normal
   week's miss, then the rows that make up the read, each tagged by what the backtests say about it:
   priced in (ESPN already counts it), too small (real, but a fraction of a target), context (descriptive,
   not tested against ESPN), role (from the Roles screen). The bottom line never edits the number. */

export const TAG_LABEL: Record<MxTag, string> = { priced: "Priced in", small: "Too small", context: "Context", role: "Role" };
export const TAG_HELP: Record<MxTag, string> = {
  priced: "Measured 2023–25: ESPN’s projection already moves with this by about this much, and nothing usable is left over.",
  small: "A real trait, but it moves a starter by a fraction of a target, far less than a game’s ordinary swing.",
  context: "Descriptive. Not tested against ESPN’s number, so it never changes the projection.",
  role: "From the Roles screen. Describes what changed; it added almost nothing in backtests once ESPN’s number was known.",
};

export const ord = (n: number) => `${n}${n % 100 >= 11 && n % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th"}`;

/** A defense's rating as a label and a place in line (1 = toughest of 32). Neutral ink: tough or soft is not good or bad until you know whose side you are on. */
export function Grade({ g }: { g?: MxGrade }) {
  if (!g) return <span className="dim">–</span>;
  const x = ((g.rank - 1) / Math.max(1, g.of - 1)) * 100;
  return (
    <span className={`mx-grade ${g.label.toLowerCase()}`} title={`${g.label}: ${ord(g.rank)} of ${g.of} (1 = toughest). ${Math.abs(g.z).toFixed(1)} standard deviations ${g.z >= 0 ? "softer" : "tougher"} than the average defense, from ${g.plays.toLocaleString()} plays this season (counted ${Math.round(g.w_now * 100)}%) blended with last season.`}>
      <b>{g.label}</b><i className="rk" aria-hidden><u style={{ left: `${x}%` }} /></i><small className="num">{g.rank}</small>
    </span>
  );
}

export const signed1 = (x: number) => { const t = Math.abs(x).toFixed(1); return Number(t) === 0 ? "0.0" : `${x > 0 ? "+" : "−"}${t}`; };

/** Where the matchup sits against a normal week: the band is ±1 typical miss of ESPN’s number, the mark is the matchup. */
export function Band({ pts, miss, wide }: { pts: number; miss: number; wide?: boolean }) {
  const span = Math.max(miss, Math.abs(pts)) * 1.08;
  const x = (v: number) => 50 + (v / span) * 50;
  return (
    <div className={`mx-band ${wide ? "wide" : ""}`} role="img" aria-label={`The matchup is worth ${signed1(pts)} points; a normal week misses by about ${miss.toFixed(1)}`}
      title={`Mark: how much of the number is this matchup (${signed1(pts)}). Band: a normal week lands within ±${miss.toFixed(1)} of ESPN’s projection.`}>
      <i className="bd" style={{ left: `${x(-miss)}%`, width: `${x(miss) - x(-miss)}%` }} />
      <i className="mid" />
      <i className="mk" style={{ left: `${x(pts)}%` }} />
    </div>
  );
}

export function TagChip({ tag }: { tag: MxTag }) {
  return <span className={`mx-tag ${tag}`} title={TAG_HELP[tag]}>{TAG_LABEL[tag]}</span>;
}

function Row({ r }: { r: MxCardRow }) {
  const one = r.key === "role"; // the role row is one sentence, not a him/them pair
  return (
    <div className={`mx-crow ${one ? "one" : ""}`} title={r.tip}>
      <span className="k">{r.label}</span>
      {one
        ? <span className="him">{r.him}{r.them ? <small>{r.them}</small> : null}</span>
        : <><span className="him">{r.him ?? <i className="dim">–</i>}</span><span className="them">{r.them ?? <i className="dim">–</i>}</span></>}
      <span className="eff num">{r.effect ?? ""}</span>
      <TagChip tag={r.tag} />
    </div>
  );
}

export function MatchupCard({ c, busy, extra, side }: { c: MxCard; busy?: boolean; extra?: ReactNode; side?: "you" | "them" }) {
  const e = c.proj.espn;
  const live = (!c.dead && e != null && !c.proj.bye) || (!!c.through && e != null);
  return (
    <section className={`mx-card ${c.bottom.tone} ${side ?? ""}`} aria-label="Start/sit card">
      <div className="mx-cardtop">
        <div className="mx-big">
          <small>{c.through ? `ESPN · week ${c.week} · his game${c.sched ? ` vs ${c.sched}` : ""}` : `ESPN · week ${c.week}`}</small>
          <b className="num">{live ? f1(e) : c.proj.bye ? "Bye" : "0.0"}</b>
          <span className="sub">
            {c.proj.model != null && <><span title="Our own forecast. For a single week ESPN’s number beat it in backtests.">Model <span className="num">{f1(c.proj.model)}</span></span> · </>}
            <Status p={{ game: { bye: c.proj.bye }, injury: c.proj.injury ?? "ACTIVE", espn_week: e }} />
          </span>
          {c.priced && c.miss_sd != null && (live || c.through) && (
            <>
              <Band pts={c.priced.pts} miss={c.miss_sd} />
              <span className="cap">{c.through ? <>vs {c.opp} it would be </> : <>matchup </>}<b className="num">{signed1(c.priced.pts)}</b>{c.through ? " in a number like this" : " of it"} · a normal week misses <b className="num">±{c.miss_sd.toFixed(1)}</b></span>
            </>
          )}
          {busy && <span className="cap">updating…</span>}
        </div>
        <div className="mx-crows" role="table" aria-label="What goes into the read">
          {c.rows.length > 0 && <div className="mx-crow hd" role="row"><span /><span>him</span><span>them</span><span className="eff">in the number</span><span /></div>}
          {c.rows.map((r) => <Row key={r.key} r={r} />)}
          {c.rows.length === 0 && <p className="dim mx-cnone">{c.bottom.headline}</p>}
        </div>
      </div>
      <div className="mx-bottom">
        <h3>{c.bottom.headline}</h3>
        {c.bottom.lines.map((l, i) => <p key={i}>{l}</p>)}
        {extra}
      </div>
    </section>
  );
}

/** Bottom-line text as Markdown, for Copy for Claude. */
export function cardMarkdown(c: MxCard): string[] {
  const out = [`**${c.bottom.headline}**`, ...c.bottom.lines.map((l) => `- ${l}`)];
  if (c.rows.length) {
    out.push("", "| Row | Him | Them | In ESPN’s number | Evidence |", "| --- | --- | --- | --- | --- |");
    for (const r of c.rows) out.push(`| ${r.label} | ${r.him ?? "–"} | ${r.them ?? "–"} | ${r.effect ?? ""} | ${TAG_LABEL[r.tag]} |`);
  }
  if (c.priced && c.miss_sd != null) out.push("", `Matchup ${sg2(c.priced.pts, 1)} of ESPN’s ${c.proj.espn?.toFixed(1)}; a normal week misses by ±${c.miss_sd.toFixed(1)}.`);
  return out;
}

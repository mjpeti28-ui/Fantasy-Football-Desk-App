import type { Pos, RoleProfile, RolesBoard, RoleTrend } from "../types";
import { useFile } from "./FileContext";
import { POS_VAR } from "./bits";
import { throughWeek } from "../format";
import "./role.css";

const UP = new Set(["snap_up", "share_up", "bigger_role"]);
const DOWN = new Set(["snap_down", "share_down", "smaller_role"]);
const dir = (t: RoleTrend) => (UP.has(t.kind) ? "up" : DOWN.has(t.kind) ? "down" : t.kind === "share_spike" ? "spike" : "flat");
const posName: Record<string, string> = { QB: "QBs", RB: "RBs", WR: "WRs", TE: "TEs" };

/** How often changes like this held, and whether ESPN's next projection already reflected them. */
export function holdText(t: RoleTrend): string {
  if (t.hold == null) return "";
  const h = Math.round(t.hold * 100);
  let s = t.season ? `Early-season ${t.kind === "bigger_role" ? "gains" : "drops"} like this held ${h}% of the time.`
    : t.cause?.length && t.hold_if_still_out != null ? `Jumps like this held ${h}% of the time; ${Math.round(t.hold_if_still_out * 100)}% when the teammate stayed out.`
    : `Changes like this held ${h}% of the time over the next two games.`;
  if (t.next_vs_espn != null && t.next_lo != null && t.next_hi != null) {
    s += t.next_lo > 0 || t.next_hi < 0
      ? ` Next week, tagged players scored ${t.next_vs_espn > 0 ? "+" : ""}${t.next_vs_espn.toFixed(1)} vs ESPN's projection: ESPN ${t.next_vs_espn < 0 ? "overrates" : "underrates"} this.`
      : " ESPN's next projection already reflects it.";
  }
  return s;
}

/** The one trend worth a glyph on a tile or row: snap changes only (they're the ones that hold). */
export function keyTrend(r?: RoleProfile | null): RoleTrend | null {
  if (!r) return null;
  return r.trends.find((t) => t.kind === "snap_up") ?? r.trends.find((t) => t.kind === "snap_down") ?? null;
}

export function TrendGlyph({ r }: { r?: RoleProfile | null }) {
  const t = keyTrend(r);
  if (!t) return null;
  const up = t.kind === "snap_up";
  return <i className={`rt-glyph ${up ? "up" : "down"}`} title={`${t.label}: ${t.text} ${holdText(t)}`} aria-label={t.label}>{up ? "▲" : "▼"}</i>;
}

/** Compact label for table rows: style, plus the key trend. */
export function RoleTag({ r }: { r?: RoleProfile | null }) {
  if (!r) return <span className="role-tag dim">No snaps yet</span>;
  return <span className="role-tag"><span className="st">{r.style?.name ?? r.tier}</span><TrendGlyph r={r} /></span>;
}

function Usage({ r, pos }: { r: RoleProfile; pos: Pos }) {
  const s = r.series;
  if (!s || !s.weeks.length) return null;
  const n = s.weeks.length;
  const W = 252, H = 74, L = 22, B = 14, bw = Math.min(26, (W - L) / n - 6);
  const x = (i: number) => L + ((W - L) / n) * (i + 0.5);
  const y = (v: number) => 4 + (1 - v / 100) * (H - B - 4);
  const line = s.snap.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  return (
    <figure className="rl-usage">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Snap share by week: ${s.snap.join(", ")}. ${s.share_label}: ${s.share.join(", ")}`}>
        {[0, 50, 100].map((v) => <g key={v}><line x1={L} x2={W} y1={y(v)} y2={y(v)} className="gl" /><text x={L - 4} y={y(v) + 3} textAnchor="end">{v}</text></g>)}
        {s.share.map((v, i) => v == null ? null : <rect key={i} x={x(i) - bw / 2} y={y(v)} width={bw} height={Math.max(1, y(0) - y(v))} rx={2} style={{ fill: POS_VAR[pos] }} className="bar"><title>{`Week ${s.weeks[i]}: ${s.share_label.toLowerCase()} ${v}%`}</title></rect>)}
        <path d={line} className="snapline" />
        {s.snap.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r={2.6} className="snapdot"><title>{`Week ${s.weeks[i]}: ${v}% of snaps`}</title></circle>)}
        {s.weeks.map((w, i) => <text key={w} x={x(i)} y={H - 2} textAnchor="middle">{w}</text>)}
      </svg>
      <figcaption><i className="k-line" /> snap % <i className="k-bar" style={{ background: POS_VAR[pos] }} /> {s.share_label.toLowerCase()} %, by week</figcaption>
    </figure>
  );
}

export function RoleSection({ r, board, pos, id, list }: { r?: RoleProfile | null; board: RolesBoard | null; pos: Pos; id?: number; list?: number[] }) {
  const { open } = useFile();
  if (pos === "K" || pos === "DST") return null;
  if (!board) return <section className="rl"><h6>Role</h6><p className="ins-note">Loading usage…</p></section>;
  if (!r) return <section className="rl"><h6>Role</h6><p className="ins-note">No snaps this season, so no usage to read.</p></section>;
  const st = r.style, sm = (st?.pff === false && board.styles_free?.[pos]) || board.styles[pos], yoy = sm?.yoy_same;
  const acc = sm?.early_acc?.[r.games < 6 ? "G3" : "G6"];
  const probs = st ? Object.entries(st.probs).sort((a, b) => b[1] - a[1]) : [];
  return (
    <section className="rl">
      <h6>Role {id != null && <button type="button" className="rl-open" onClick={() => open(id, list)}>Open player file ⤢</button>}</h6>
      <div className="rl-hd">
        <b>{st ? (st.p >= 0.7 ? st.name : `${st.name}, or possibly ${st.alt.replace(/\b(?![A-Z]{2,}\b)\w+/g, (w) => w.toLowerCase())}`) : "Not enough usage to type"}</b>
        <span className="tier">{r.tier}</span>
      </div>
      {st && (
        <div className="rl-mix" title={probs.map(([k, v]) => `${k} ${Math.round(v * 100)}%`).join(" · ")} aria-label="Style mix">
          {probs.filter(([, v]) => v >= 0.03).map(([k, v], i) => <i key={k} className={i === 0 ? "top" : ""} style={{ width: `${v * 100}%`, ...(i === 0 ? { background: POS_VAR[pos] } : {}) }} />)}
        </div>
      )}
      {r.route_sh && <p className="ins-note" title="Routes run (PFF) divided by his team's dropbacks in the games he played. Descriptive: it says how much of the passing game he's part of, not a forecast.">Runs a route on {r.route_sh.v}% of dropbacks · {r.route_sh.rank} of {r.route_sh.of} {posName[pos]} (PFF)</p>}
      {r.traits.length > 0 && (
        <div className="rl-traits">
          {r.traits.map((t) => <span key={t.label} className="chip" tabIndex={0} title={`${t.text}, among ${posName[pos]} ${board.fit_seasons[0]}–${String(board.fit_seasons[1]).slice(2)}. Shown only once the number is mostly signal.`}>{t.label}</span>)}
        </div>
      )}
      {r.trends.length > 0 && (
        <ul className="rl-trends">
          {r.trends.map((t) => (
            <li key={t.kind} className={dir(t)}>
              <span className="ic" aria-hidden="true">{dir(t) === "up" ? "▲" : dir(t) === "down" ? "▼" : dir(t) === "spike" ? "◆" : "•"}</span>
              <span className="tx"><b>{t.label}</b> {t.text}
                {t.hold != null && <span className={`hold ${t.hold >= 0.6 ? "real" : "weak"}`}>{holdText(t)}</span>}</span>
            </li>
          ))}
        </ul>
      )}
      <Usage r={r} pos={pos} />
      <p className="ins-note">
        {r.games} game{r.games === 1 ? "" : "s"} this season; small samples are pulled toward last season.
        {acc && r.games < 10 ? ` Style calls this early match the full-season call ${Math.round(acc.acc * 100)}% of the time.` : ""}
        {yoy != null ? ` Players keep the same style year to year ${Math.round(yoy * 100)}% of the time.` : ""}
      </p>
    </section>
  );
}

/** "What changed" on a roster: every player with a usage trend, strongest first. */
export function RosterShifts({ players, board, onSelect, title = "Usage shifts on your roster" }: {
  players: { id: number; name: string; pos: Pos }[]; board: RolesBoard | null; onSelect: (id: number) => void; title?: string;
}) {
  if (!board) return null;
  const rank: Record<string, number> = { snap_up: 0, snap_down: 1, bigger_role: 2, smaller_role: 3, share_up: 4 };
  const rows = players.map((p) => ({ p, ts: (board.players[String(p.id)]?.trends ?? []).filter((t) => t.kind in rank).sort((a, b) => rank[a.kind] - rank[b.kind]) }))
    .filter((r) => r.ts.length).sort((a, b) => rank[a.ts[0].kind] - rank[b.ts[0].kind]).map(({ p, ts }) => ({ p, t: ts[0], more: ts.slice(1) }));
  return (
    <section className="rl">
      <h6>{title} · {throughWeek(board.through_week, board.partial)}</h6>
      {!rows.length ? <p className="ins-note">No snap or role changes big enough to flag.</p> : (
        <ul className="rl-shifts">
          {rows.map(({ p, t, more }) => (
            <li key={`${p.id}-${t.kind}`}>
              <button type="button" onClick={() => onSelect(p.id)} title={`${t.text} ${holdText(t)}`}>
                <span className={`ic ${dir(t)}`} aria-hidden="true">{dir(t) === "up" ? "▲" : dir(t) === "down" ? "▼" : "◆"}</span>
                <span className="who">{p.name}</span>
                <span className="what">{t.label}{t.cause?.length ? `, ${t.cause.join(", ")} out` : ""}{more.length ? ` · ${more.map((m) => m.label.toLowerCase()).join(" · ")}` : ""}</span>
                <span className={`h ${t.hold != null && t.hold >= 0.6 ? "real" : "weak"}`}>{t.hold != null ? `${Math.round(t.hold * 100)}%` : ""}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {rows.length > 0 && <p className="ins-note">% = how often the first change held up, 2021–25. Target spikes without more snaps are left out here: they mostly fade.</p>}
    </section>
  );
}

import type { ReactNode } from "react";
import type { Player, PowerTeam } from "../types";
import { f1, gameLabel, pct, signed } from "../format";
import { Face, Spark, Status } from "./bits";
import { CompareBtn } from "./FileContext";

export interface MoveOpt { label: string; to: string; here: boolean }

export function PlayerInspector({ p, onCopy, moves, onMove, h2h, onLog, role, signals }: { p: Player; onCopy: () => void; moves: MoveOpt[]; onMove: (to: string) => void; h2h: ReactNode; onLog?: () => void; role?: ReactNode; signals?: ReactNode }) {
  const g = p.game;
  return (
    <div className="ins">
      <div className="ins-hd"><Face p={p} /><div><div className="nm">{p.name}</div><div className="dim">{p.pos} · {p.team} · {gameLabel(g, true, g?.day, g?.time)}</div></div></div>
      {moves.length > 0 && <section>
        <h6>Move to</h6>
        <div className="moves">{moves.map((m) => <button key={m.to} type="button" className={m.here ? "here" : ""} disabled={m.here} onClick={() => onMove(m.to)}>{m.label}</button>)}</div>
      </section>}
      <section>
        <h6>This week</h6>
        <div className="ins-h2h">{h2h}</div>
        <dl className="kv">
          <dt>ESPN projection</dt><dd>{f1(p.espn_week)}</dd>
          <dt>Team total (Vegas)</dt><dd>{f1(g?.implied_total ?? null)}</dd>
          <dt>Chance he plays</dt><dd>{pct(p.avail_now)}</dd>
          <dt>Status</dt><dd><Status p={p} /></dd>
        </dl>
      </section>
      {role}
      {signals}
      <section>
        <h6>Season</h6>
        <dl className="kv">
          <dt>Points per game</dt><dd>{f1(p.ppg)} in {p.games}</dd>
          <dt>Weeks 1–{p.weekly.length}</dt><dd><Spark w={p.weekly} from={1} /></dd>
          <dt>Rest of season</dt><dd>{f1(p.est_ppg)} ppg{["QB", "RB", "WR", "TE"].includes(p.pos) ? " ±4" : ""}</dd>
          <dt>Experts' rank</dt><dd>{p.ecr_rank ? `${p.pos}${Math.round(p.ecr_rank)}` : "–"}</dd>
          <dt>Plays rest of season</dt><dd>{pct(p.avail_ros)}</dd>
          <dt>Bye</dt><dd>{p.bye ? `Week ${p.bye}` : "–"}</dd>
        </dl>
      </section>
      <div className="ins-acts"><CompareBtn id={p.id} /><button className="tbtn" type="button" onClick={onCopy}>Copy for Claude</button>{onLog && <button className="tbtn" type="button" onClick={onLog}>Log this call…</button>}</div>
      <p className="ins-note">Rest of season = average of our prior-plus-results, ESPN, and the experts' consensus. Typical miss about ±4 ppg.</p>
    </div>
  );
}

export function TeamInspector({ t, isMe, isOpp, shifts }: { t: PowerTeam; isMe: boolean; isOpp: boolean; shifts?: ReactNode }) {
  return (
    <div className="ins">
      <div><div className="nm" style={{ fontSize: 17, fontWeight: 600 }}>{t.name}</div><div className="dim">{isMe ? "Your team" : isOpp ? "This week's opponent" : "League"} · {t.record.w}–{t.record.l}</div></div>
      <section>
        <h6>So far</h6>
        <dl className="kv">
          <dt>Points</dt><dd>{f1(t.pf)} (#{t.pf_rank})</dd>
          <dt>Projected for lineups used</dt><dd>{f1(t.proj_pf)}</dd>
          <dt>Above projection</dt><dd style={{ color: (t.luck ?? 0) >= 0 ? "var(--above)" : "var(--below)" }}>{signed(t.luck)}</dd>
          <dt>Weekly</dt><dd><span className="wk"><Spark w={t.weekly} bw={7} h={18} from={1} /><small className="dim num">{t.weekly.map((x) => Math.round(x)).join(" ")}</small></span></dd>
        </dl>
      </section>
      {shifts}
      {t.odds && <section>
        <h6>Rest of season</h6>
        <dl className="kv">
          <dt>Projected final points</dt><dd>{f1(t.proj_final_pts)}</dd>
          <dt>Top-2 seed</dt><dd>{pct(t.odds.top2)}</dd>
          <dt>Playoffs</dt><dd>{pct(t.odds.playoffs)}</dd>
          <dt>Title</dt><dd className="dim">{pct(t.odds.title)}</dd>
        </dl>
      </section>}
      <p className="ins-note">Seeding is by total points. Above-projection points count for seeding but haven't carried over in past seasons. Title odds are close to a lottery.</p>
    </div>
  );
}

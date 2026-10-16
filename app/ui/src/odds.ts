import type { Player } from "./types";
import { isDead } from "./components/bits";

/** The few fields the odds need, so a card for someone not on the roster can be compared too. */
export type Pt = Pick<Player, "pos" | "espn_week" | "injury" | "game">;

/** P(a outscores b this week). Each player's outcome = ESPN projection × the 2021-25 spread of
 *  actual/forecast at his position (quantile grid), treated as independent. Approximate: same-game
 *  correlation is ignored and the ratios were measured against our forecast, not ESPN's. */
export function pBeats(a: Pt, b: Pt, q?: Record<string, number[]>): number | null {
  if (isDead(b) && !isDead(a)) return 1;
  if (isDead(a)) return 0;
  const qa = q?.[a.pos], qb = q?.[b.pos];
  if (!qa || !qb || !a.espn_week || !b.espn_week) return null;
  let win = 0;
  for (const ra of qa) for (const rb of qb) {
    const x = ra * a.espn_week, y = rb * b.espn_week;
    win += x > y ? 1 : x === y ? 0.5 : 0;
  }
  return win / (qa.length * qb.length);
}

/** P(a scores at least `x` this week), and P(under `x`): the same quantile grid applied to one player. */
export function pAtLeast(a: Pt, x: number, q?: Record<string, number[]>): number | null {
  const qa = q?.[a.pos];
  if (!qa || !a.espn_week) return null;
  return qa.filter((r) => r * a.espn_week! >= x).length / qa.length;
}
export const pUnder = (a: Pt, x: number, q?: Record<string, number[]>): number | null => { const v = pAtLeast(a, x, q); return v == null ? null : 1 - v; };

/** Normal CDF (Abramowitz-Stegun 7.1.26), for win chance from a projected margin. */
export function phi(z: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2), y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-(z * z) / 2);
  return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

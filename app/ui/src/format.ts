export const f1 = (x: number | null | undefined) =>
  x == null || Number.isNaN(x) ? "–" : (x < 0 ? "−" : "") + Math.abs(x).toFixed(1);

export const signed = (x: number | null | undefined) =>
  x == null ? "–" : (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(1);

export const pct = (x: number | null | undefined) =>
  x == null ? "–" : x >= 0.995 ? ">99%" : x < 0.005 ? "<1%" : `${Math.round(x * 100)}%`;

// "through week 5 (week 6 under way)" when only some of the latest week's games are in
export const throughWeek = (w: number, partial?: boolean) => (partial ? `through week ${w - 1} (week ${w} under way)` : `through week ${w}`);
/** One short form per team, used wherever a full team name won't fit: the whole name if short, else its last word
 * ("Stinkers", "Gemstones") when that's unique in the league, else its first two words. Leading "The" is dropped. */
export function teamShort(name: string, all: string[]): string {
  const strip = (n: string) => n.replace(/^the\s+/i, "").trim();
  const n = strip(name);
  if (n.length <= 12) return n;
  const last = (x: string) => strip(x).split(/\s+/).pop()!.toLowerCase();
  const w = n.split(/\s+/), lw = w[w.length - 1];
  if (lw.length >= 4 && /[a-z]/i.test(lw) && all.filter((x) => last(x) === lw.toLowerCase()).length <= 1) return lw;
  return w.slice(0, 2).join(" ");
}

/** Whole percents that add to 100 (largest remainder), so a 62/38 split never reads 62/39. */
export function wholePcts(vals: number[]): number[] {
  const tot = vals.reduce((a, v) => a + v, 0) || 1, raw = vals.map((v) => (100 * v) / tot), out = raw.map(Math.floor);
  const order = raw.map((r, i) => [r - Math.floor(r), i] as const).sort((a, b) => b[0] - a[0]);
  for (let k = 0; k < 100 - out.reduce((a, v) => a + v, 0); k++) out[order[k % order.length][1]]++;
  return out;
}

/** "Christian Watson" -> "C. Watson" once a full name is too long for a tight row; defenses and short names stay whole. */
export const tightName = (n: string, max = 14) => (n.length <= max || n.endsWith("D/ST") || !n.includes(" ") ? n : `${n[0]}. ${n.split(" ").slice(1).join(" ")}`);

export const lastName = (n: string) => (n.endsWith("D/ST") ? n : n.split(" ").slice(1).join(" ") || n);

export function clock(t?: string) {
  if (!t) return "";
  const [h, m] = t.split(":").map(Number);
  return `${((h + 11) % 12) + 1}:${String(m).padStart(2, "0")}${h < 12 ? "a" : "p"}`;
}

export function gameLabel(g: { bye: boolean; home?: boolean; opp?: string } | null, withTime = false, day?: string, time?: string) {
  if (!g) return "–";
  if (g.bye) return "Bye";
  return `${g.home ? "vs" : "@"} ${g.opp}${withTime && day ? ` · ${day} ${clock(time)}` : ""}`;
}

export function age(now: number, t: number | null | undefined) {
  if (!t) return "unknown";
  const h = (now - t) / 3600;
  if (h < 1) return "just now";
  if (h < 24) return `${Math.round(h)}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

// Standard normal CDF (Abramowitz–Stegun 7.1.26), good to ~1e-7.
export function phi(z: number) {
  const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-(z * z) / 2);
  return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

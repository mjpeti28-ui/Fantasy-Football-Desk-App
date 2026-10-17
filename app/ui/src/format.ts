export const f1 = (x: number | null | undefined) =>
  x == null || Number.isNaN(x) ? "–" : (x < 0 ? "−" : "") + Math.abs(x).toFixed(1);

export const signed = (x: number | null | undefined) =>
  x == null ? "–" : (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(1);

export const pct = (x: number | null | undefined) =>
  x == null ? "–" : x >= 0.995 ? ">99%" : x < 0.005 ? "<1%" : `${Math.round(x * 100)}%`;

// "through week 5 (week 6 under way)" when only some of the latest week's games are in
export const throughWeek = (w: number, partial?: boolean) => (partial ? `through week ${w - 1} (week ${w} under way)` : `through week ${w}`);
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

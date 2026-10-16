/** Team colors for roster tiles, made contrast-safe. Colors come from ESPN's teams endpoint
 *  (shipped by the engine as `team_colors`). */
export type ColorMode = "position" | "stripe" | "tint" | "card";
export interface TeamColor { name: string; color: string; alt: string }
export interface TilePalette { sig: string; second: string; bg: string; accent: string; hi: string; lo: string }

const rgb = (h: string) => { const n = parseInt(h.replace("#", ""), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };
const hex = (c: number[]) => "#" + c.map((v) => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, "0")).join("");
const lin = (v: number) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
export const lum = (h: string) => { const [r, g, b] = rgb(h); return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b); };
export const contrast = (a: string, b: string) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
export const mix = (a: string, b: string, t: number) => { const A = rgb(a), B = rgb(b); return hex(A.map((v, i) => v * t + B[i] * (1 - t))); };

/** sig: the color that identifies the team and stays visible on the tile surface (falls back to
 *  the alternate when the primary is near the surface, e.g. Raiders/Steelers black in dark mode).
 *  bg: a deep version of the primary that keeps white text at ≥ 5:1. */
export function tilePalette(abbr: string | null, colors: Record<string, TeamColor> | undefined, dark: boolean): TilePalette | null {
  const c = abbr ? colors?.[abbr] : undefined;
  if (!c) return null;
  const surface = dark ? "#2a2d2b" : "#fcfcfa";
  const sig = contrast(c.color, surface) >= 1.6 ? c.color : c.alt;
  const second = sig === c.color ? c.alt : c.color;
  let bg = c.color, t = 1;
  while (contrast("#ffffff", bg) < 5 && t > 0.2) { t -= 0.05; bg = mix(c.color, "#0c0d0e", t); }
  if (lum(c.color) < 0.012) bg = mix(c.alt, "#0c0d0e", 0.28); // pure-black teams: hint of the alternate
  // accent: the team's second color, used on the tile itself, so it must stand off the deep bg
  // (Bengals/Panthers/Saints list black as the alternate: fall back to a light primary).
  let accent = c.alt;
  if (contrast(accent, bg) < 2.2) accent = contrast(c.color, bg) >= 2.2 ? c.color : mix(c.color, "#ffffff", 0.45);
  return {
    sig, second: contrast(second, sig) < 1.3 ? (dark ? "#ffffff" : "#1c211d") : second, bg, accent,
    hi: mix(bg, "#ffffff", 0.93), lo: mix(bg, "#000000", 0.86),   // gradient stops for the bevel
  };
}

/** Grey stand-in for players who can't score (keeps the same shape as a team palette). */
export const DEAD_PALETTE = (dark: boolean): TilePalette =>
  ({ sig: "#6b6f6c", second: "#8a8e8b", bg: dark ? "#3a3d3b" : "#5d615e", accent: "#9a9e9b", hi: dark ? "#4a4d4b" : "#6d716e", lo: dark ? "#2a2c2b" : "#4a4d4b" });

/* Small, dependency-free numerics for the Lab: z-scores, k-means (k-means++ with restarts), silhouette, resample
   stability (adjusted Rand), a 2-component PCA for the map, nearest neighbours and Spearman correlation.
   Sizes are a few hundred players by a handful of stats, so plain O(n²) loops are fine. */

export type Mat = number[][];

/** Seeded PRNG (mulberry32), so the same inputs give the same clusters. */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

export function zscore(X: Mat): { Z: Mat; mu: number[]; sd: number[] } {
  const n = X.length, d = X[0]?.length ?? 0;
  const mu = Array.from({ length: d }, (_, j) => X.reduce((s, r) => s + r[j], 0) / Math.max(1, n));
  const sd = Array.from({ length: d }, (_, j) => Math.sqrt(X.reduce((s, r) => s + (r[j] - mu[j]) ** 2, 0) / Math.max(1, n - 1)) || 1);
  return { Z: X.map((r) => r.map((v, j) => (v - mu[j]) / sd[j])), mu, sd };
}

const d2 = (a: number[], b: number[]) => { let s = 0; for (let i = 0; i < a.length; i++) { const t = a[i] - b[i]; s += t * t; } return s; };

export interface KM { labels: number[]; centers: Mat; inertia: number }

function once(Z: Mat, k: number, rand: () => number): KM {
  const n = Z.length;
  const centers: Mat = [Z[Math.floor(rand() * n)].slice()];
  const best = Z.map((z) => d2(z, centers[0]));
  while (centers.length < k) { // k-means++: next center drawn in proportion to squared distance
    const tot = best.reduce((s, v) => s + v, 0);
    let r = rand() * tot, i = 0;
    while (i < n - 1 && r > best[i]) { r -= best[i]; i++; }
    centers.push(Z[i].slice());
    for (let j = 0; j < n; j++) best[j] = Math.min(best[j], d2(Z[j], Z[i]));
  }
  const labels = new Array(n).fill(0);
  for (let it = 0; it < 100; it++) {
    let moved = 0;
    for (let i = 0; i < n; i++) {
      let bi = 0, bd = Infinity;
      for (let c = 0; c < k; c++) { const dd = d2(Z[i], centers[c]); if (dd < bd) { bd = dd; bi = c; } }
      if (labels[i] !== bi) { labels[i] = bi; moved++; }
    }
    for (let c = 0; c < k; c++) {
      const m = Z.filter((_, i) => labels[i] === c);
      if (!m.length) { // empty cluster: restart it on the point furthest from its center
        let fi = 0, fd = -1;
        for (let i = 0; i < n; i++) { const dd = d2(Z[i], centers[labels[i]]); if (dd > fd) { fd = dd; fi = i; } }
        centers[c] = Z[fi].slice(); labels[fi] = c; moved++; continue;
      }
      centers[c] = m[0].map((_, j) => m.reduce((s, r) => s + r[j], 0) / m.length);
    }
    if (!moved && it > 0) break;
  }
  return { labels, centers, inertia: Z.reduce((s, z, i) => s + d2(z, centers[labels[i]]), 0) };
}

/** Best of several k-means++ starts; clusters renumbered largest first so colours stay put between runs. */
export function kmeans(Z: Mat, k: number, seed = 7, restarts = 12): KM {
  const rand = rng(seed);
  let best: KM | null = null;
  for (let r = 0; r < restarts; r++) { const m = once(Z, k, rand); if (!best || m.inertia < best.inertia) best = m; }
  const b = best!;
  const size = b.centers.map((_, c) => b.labels.filter((l) => l === c).length);
  const order = size.map((s, c) => [s, c]).sort((u, v) => v[0] - u[0]).map(([, c]) => c);
  const re = new Map(order.map((c, i) => [c, i]));
  return { labels: b.labels.map((l) => re.get(l)!), centers: order.map((c) => b.centers[c]), inertia: b.inertia };
}

/** Mean silhouette: +1 = tight, well-separated groups; ~0 = groups that blur into each other. */
export function silhouette(Z: Mat, labels: number[], k: number): number {
  const n = Z.length;
  if (n < 3 || k < 2) return 0;
  const D = Z.map((a) => Z.map((b) => Math.sqrt(d2(a, b))));
  let s = 0;
  for (let i = 0; i < n; i++) {
    const sum = new Array(k).fill(0), cnt = new Array(k).fill(0);
    for (let j = 0; j < n; j++) if (j !== i) { sum[labels[j]] += D[i][j]; cnt[labels[j]]++; }
    if (!cnt[labels[i]]) continue;
    const a = sum[labels[i]] / cnt[labels[i]];
    let b = Infinity;
    for (let c = 0; c < k; c++) if (c !== labels[i] && cnt[c]) b = Math.min(b, sum[c] / cnt[c]);
    if (Number.isFinite(b)) s += (b - a) / Math.max(a, b);
  }
  return s / n;
}

/** Adjusted Rand index between two labelings of the same points (1 = identical groups, ~0 = chance). */
export function ari(a: number[], b: number[]): number {
  const n = a.length, key = (x: number, y: number) => `${x},${y}`;
  const nij = new Map<string, number>(), ai = new Map<number, number>(), bj = new Map<number, number>();
  for (let i = 0; i < n; i++) { nij.set(key(a[i], b[i]), (nij.get(key(a[i], b[i])) ?? 0) + 1); ai.set(a[i], (ai.get(a[i]) ?? 0) + 1); bj.set(b[i], (bj.get(b[i]) ?? 0) + 1); }
  const c2 = (x: number) => (x * (x - 1)) / 2;
  const sij = [...nij.values()].reduce((s, v) => s + c2(v), 0), sa = [...ai.values()].reduce((s, v) => s + c2(v), 0), sb = [...bj.values()].reduce((s, v) => s + c2(v), 0);
  const exp = (sa * sb) / c2(n), mx = (sa + sb) / 2;
  return mx === exp ? 1 : (sij - exp) / (mx - exp);
}

/** Re-cluster random 80% subsamples and compare with the full-data groups on the same players (mean ARI). */
export function stability(Z: Mat, k: number, ref: number[], runs = 10, seed = 11): number {
  const rand = rng(seed), out: number[] = [];
  for (let r = 0; r < runs; r++) {
    const idx = Z.map((_, i) => i).filter(() => rand() < 0.8);
    if (idx.length <= k * 2) continue;
    const m = kmeans(idx.map((i) => Z[i]), k, seed + r + 1, 4);
    out.push(ari(m.labels, idx.map((i) => ref[i])));
  }
  return out.length ? out.reduce((s, v) => s + v, 0) / out.length : 0;
}

/** First two principal components of z-scored data: coordinates, loadings and share of variance each explains. */
export function pca2(Z: Mat): { xy: [number, number][]; load: [number[], number[]]; ve: [number, number] } {
  const n = Z.length, d = Z[0]?.length ?? 0;
  const C = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => Z.reduce((s, r) => s + r[i] * r[j], 0) / Math.max(1, n - 1)));
  const tot = C.reduce((s, r, i) => s + r[i], 0) || 1;
  const comps: number[][] = [], vals: number[] = [];
  const M = C.map((r) => r.slice());
  for (let c = 0; c < Math.min(2, d); c++) {
    let v = Array.from({ length: d }, (_, i) => 1 / Math.sqrt(d) + 0.01 * i);
    let lam = 0;
    for (let it = 0; it < 300; it++) {
      const w = M.map((r) => r.reduce((s, x, j) => s + x * v[j], 0));
      const nr = Math.sqrt(w.reduce((s, x) => s + x * x, 0)) || 1;
      v = w.map((x) => x / nr); lam = nr;
    }
    // sign convention: the largest loading points right/up, so the map doesn't flip between runs
    const big = v.reduce((bi, x, i) => (Math.abs(x) > Math.abs(v[bi]) ? i : bi), 0);
    if (v[big] < 0) v = v.map((x) => -x);
    comps.push(v); vals.push(lam);
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) M[i][j] -= lam * v[i] * v[j];
  }
  while (comps.length < 2) { comps.push(new Array(d).fill(0)); vals.push(0); }
  const xy = Z.map((r) => [r.reduce((s, x, j) => s + x * comps[0][j], 0), r.reduce((s, x, j) => s + x * comps[1][j], 0)] as [number, number]);
  return { xy, load: [comps[0], comps[1]], ve: [vals[0] / tot, vals[1] / tot] };
}

export function neighbors(Z: Mat, i: number, m = 6): { j: number; d: number }[] {
  return Z.map((z, j) => ({ j, d: Math.sqrt(d2(z, Z[i])) })).filter((x) => x.j !== i).sort((a, b) => a.d - b.d).slice(0, m);
}

const ranks = (v: number[]) => {
  const o = v.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]);
  const r = new Array(v.length);
  for (let i = 0; i < o.length;) { let j = i; while (j + 1 < o.length && o[j + 1][0] === o[i][0]) j++; for (let t = i; t <= j; t++) r[o[t][1]] = (i + j) / 2; i = j + 1; }
  return r as number[];
};

/** Spearman rank correlation over pairs where both values exist. */
export function spearman(a: (number | null)[], b: (number | null)[]): { r: number; n: number } | null {
  const x: number[] = [], y: number[] = [];
  for (let i = 0; i < a.length; i++) if (a[i] != null && b[i] != null && Number.isFinite(a[i]!) && Number.isFinite(b[i]!)) { x.push(a[i]!); y.push(b[i]!); }
  const n = x.length;
  if (n < 10) return null;
  const rx = ranks(x), ry = ranks(y), m = (n - 1) / 2;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { sxy += (rx[i] - m) * (ry[i] - m); sxx += (rx[i] - m) ** 2; syy += (ry[i] - m) ** 2; }
  return sxx && syy ? { r: sxy / Math.sqrt(sxx * syy), n } : null;
}

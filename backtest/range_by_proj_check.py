"""Follow-up checks for range_by_proj.py: is 2026 weeks 1-6 enough to judge it (player-cluster bootstrap of pinball
gain, weeks 1-6 of every season), and does tiering change head-to-head odds (same-team same-position pairs, >=1 starter)?

    python backtest/range_by_proj_check.py
"""
import sys,io,contextlib; from pathlib import Path; sys.path.insert(0,str(Path(__file__).parent))
import numpy as np, pandas as pd
import range_by_proj as rp
with contextlib.redirect_stdout(io.StringIO()): O=rp.main(3)
S=O[O.start].copy()
S["d"]=S.pb_tier-S.pb_pooled
# 1) weeks 1-6 in each past season: does tiering show its gain early?
print("starters wks1-6 by season: coverage pooled/tier, pinball diff (neg=tier better)")
E=S[S.week<=6]
print(E.groupby("season").agg(inp=("in_pooled","mean"),intr=("in_tier","mean"),d=("d","mean"),n=("d","size")).round(3))
# player-cluster bootstrap of pinball diff per season, wks 1-6 and full
rng=np.random.default_rng(1)
def boot(g,B=2000):
    pl=g.groupby("espn_id").d.agg(["sum","size"]); s=pl["sum"].to_numpy(); n=pl["size"].to_numpy()
    idx=rng.integers(0,len(pl),(B,len(pl))); return np.percentile(s[idx].sum(1)/n[idx].sum(1),[2.5,97.5])
for y in range(2021,2027):
    g=E[E.season==y]; print(y,"wk1-6 d=%.3f CI"%g.d.mean(),boot(g).round(3))
# 2) top tier 2026: miss side
T=S[(S.tier==2)]
T=T.assign(low=T.r<T.groupby(["season","position"]).r.transform(lambda x:0)) 
for y in [2023,2024,2025,2026]:
    g=S[(S.season==y)&(S.tier==2)&(S.week<=6)]
    print(y,"top tier wk1-6 mean ratio %.3f  median %.3f  sd %.3f n %d"%(g.r.mean(),g.r.median(),g.r.std(),len(g)))

# ---- head-to-head
D=pd.concat([rp.rows(y) for y in range(2021,2027)])
rng=np.random.default_rng(3); QG=np.linspace(0.01,0.99,99)
res=[]
for ty in range(2021,2027):
    tr=D[(D.season<2026)&(D.season!=ty)&D.week.between(3,14)]; te=D[D.season==ty]
    for p in rp.POS:
        trp=tr[tr.position==p]; cuts=np.quantile(trp.espn_proj,[1/3,2/3])
        pooled=np.quantile(trp.r,QG); tq={t:np.quantile(g.r,QG) for t,g in trp.groupby(np.searchsorted(cuts,trp.espn_proj))}
        g=te[te.position==p].copy(); g["tier"]=np.searchsorted(cuts,g.espn_proj)
        # same fantasy team, same week, same position, at least one started
        team=[c for c in g.columns if c in("team_id","fantasy_team","owner_id")][0]
        for (wk,tm),h in g.groupby(["week",team]):
            h=h.reset_index(drop=True)
            for i in range(len(h)):
                for j in range(i+1,len(h)):
                    a,b=h.loc[i],h.loc[j]
                    if not(a.start or b.start): continue
                    if a.points==b.points: continue
                    def pb(qa,qb):
                        xa=a.espn_proj*qa[:,None]; xb=b.espn_proj*qb[None,:]; return (xa>xb).mean()
                    res.append(dict(season=ty,pos=p,y=float(a.points>b.points),
                        pp=pb(pooled,pooled),pt=pb(tq[a.tier],tq[b.tier]),gap=abs(a.espn_proj-b.espn_proj)))
R=pd.DataFrame(res)
R["bp"]=(R.pp-R.y)**2; R["bt"]=(R.pt-R.y)**2
ll=lambda p,y: -(y*np.log(np.clip(p,1e-3,1-1e-3))+(1-y)*np.log(np.clip(1-p,1e-3,1-1e-3)))
R["lp"]=ll(R.pp,R.y); R["lt"]=ll(R.pt,R.y)
print(R.groupby("season")[["bp","bt","lp","lt"]].mean().round(4).assign(n=R.groupby("season").size(),moved=R.groupby("season").apply(lambda g:(g.pt-g.pp).abs().mean()).round(3)))
R["fav"]=np.maximum(R.pp,1-R.pp)
print("mean |change| in odds, pp:", ((R.pt-R.pp).abs().mean()*100).round(2), " share of pairs where favourite flips:", ((R.pp>.5)!=(R.pt>.5)).mean().round(4))
for b in [(0.5,.6),(.6,.7),(.7,.8),(.8,1)]:
    m=(R.season<2026)&R.fav.between(*b); fy=np.where(R.pp>.5,R.y,1-R.y)
    ft=np.where(R.pp>.5,R.pt,1-R.pt)
    print(b,"said pooled %.3f tier %.3f actual %.3f n %d"%(R.fav[m].mean(),ft[m].mean(),fy[m].mean(),m.sum()))

import sys; sys.argv = sys.argv if len(sys.argv) > 1 else sys.exit('usage: midseason_hold.py NFL_DIR [ACT_2026_DIR]')
sys.path.insert(0,'backtest'); import midseason_fantasy as M, pandas as pd, numpy as np
rng=np.random.default_rng(4); P=[]
for y in M.HIST:
  w=M.weekly(y); a=M.features(w[w.week<=9]); b=M.features(w[(w.week>=10)&(w.week<=17)])
  a=a[(a.games>=4)&M.relevant(y,a,9)]; b=b[b.games>=3]
  j=a.join(b.drop(columns=['position','name','espn_id']),rsuffix='_2',how='inner'); j['season']=y; P.append(j)
D=pd.concat(P)
out=[]
for pos,stats in M.STATS.items():
  d=D[D.position==pos]
  for s in stats:
    z=d[[s,s+'_2']].dropna(); r=lambda q:q[s].corr(q[s+'_2'])
    b=[r(z.iloc[rng.integers(0,len(z),len(z))]) for _ in range(500)]
    out.append((pos,s,len(z),r(z),*np.percentile(b,[2.5,97.5])))
    print('%s %-12s n%3d r=%.2f [%.2f,%.2f]'%out[-1])
pd.DataFrame(out,columns=['pos','stat','n','r','lo','hi']).to_csv('history/backtest/midseason_hold.csv',index=False)

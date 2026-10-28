import random
from analyze import interval_index,enclosing,subtract
from bisect import bisect_right
rng=random.Random(284)
for case in range(1000):
 parents=[dict(start_ns=a,end_ns=a+rng.randrange(1,100)) for a in [rng.randrange(200) for _ in range(30)]]
 child=dict(start_ns=rng.randrange(250),end_ns=0);child['end_ns']=child['start_ns']+rng.randrange(1,20)
 expected=[r for r in parents if r['start_ns']<=child['start_ns'] and child['end_ns']<=r['end_ns']]
 actual=enclosing(interval_index(parents),child)
 assert (actual is None)==(not expected)
 if expected:assert actual['end_ns']-actual['start_ns']==min(r['end_ns']-r['start_ns'] for r in expected)
 spans=[(r['start_ns'],r['end_ns']) for r in parents];roots=[(30,180)];blocked=[(60,80),(100,110)]
 clipped=subtract(subtract(spans,subtract(spans,roots)),blocked);ends=[b for a,b in clipped]
 old=sum(bool(subtract(subtract([s],subtract([s],roots)),blocked)) for s in spans)
 new=sum(i<len(clipped) and clipped[i][0]<b for a,b in spans for i in [bisect_right(ends,a)])
 assert old==new
print('Random nested interval ownership and active-call equivalence: 1000 cases PASS')

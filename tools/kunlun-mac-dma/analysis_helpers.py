import json,re,statistics as st,collections

def summary(values):
 v=list(values)
 if not v:raise ValueError('empty measurements')
 return dict(n=len(v),mean=st.mean(v),median=st.median(v),stddev=st.stdev(v) if len(v)>1 else 0,min=min(v),max=max(v),samples=v)

def bench_rows(text):
 rows=[]
 for match in re.finditer(r'\{\s*"build_commit"',text):
  try:row=json.JSONDecoder().raw_decode(text[match.start():])[0]
  except ValueError:continue
  if isinstance(row,dict) and 'samples_ns' in row:rows.append(row)
 return rows

def ticks_by_name(text):
 groups=collections.defaultdict(list)
 for line in text.splitlines():
  if line.startswith('TICKS '):groups[line.split()[1]].append(json.loads(line[line.index('['):]))
 return dict(groups)

def values(body):return dict(re.findall(r'(\w+)=([^\s]+)',body))

def parse_standard(name,text):
 match=re.fullmatch(r'std-(tiny|7b)-(pd0|pd1|layer|row)-u(\d+)-a([01])-m([01])-i(\d+)',name)
 if not match:return []
 model,split,u,share,mac,order=match.groups();rows=bench_rows(text)
 if len(rows)!=2:raise ValueError((name,'expected PP/TG rows',len(rows)))
 result=[]
 for row in rows:
  phase='pp' if row['n_prompt'] else 'tg';length=row['n_prompt'] or row['n_gen'];times=[float(t)/1e6 for t in row['samples_ns']]
  result.append(dict(name=name,model=model,split=split,ubatch=int(u),share=int(share),mac=int(mac),order=int(order),phase=phase,length=length,ms=summary(times),ts=summary(length*1000/t for t in times),raw=row))
 return result

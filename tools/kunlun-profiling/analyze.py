"""Offline analysis. Run after all measurement jobs, never alongside benchmarks.

Wall time is partitioned by interval union. Inclusive API/worker/kernel times
are separate tables and must not be added to the partition.
"""
from pathlib import Path
from collections import defaultdict
import csv,json,re,statistics,sys,math
from bisect import bisect_left,bisect_right

def union(spans):
    result=[]
    for a,b in sorted(spans):
        if b<=a:continue
        if result and a<=result[-1][1]:result[-1]=(result[-1][0],max(b,result[-1][1]))
        else:result.append((a,b))
    return result
def length(spans):return sum(b-a for a,b in union(spans))
def subtract(spans,blocked):
    a=union(spans);b=union(blocked);out=[];j=0
    for start,end in a:
        while j<len(b) and b[j][1]<=start:j+=1
        k=j;cursor=start
        while k<len(b) and b[k][0]<end:
            lo,hi=b[k]
            if lo>cursor:out.append((cursor,min(lo,end)))
            cursor=max(cursor,hi)
            if cursor>=end:break
            k+=1
        if cursor<end:out.append((cursor,end))
    return out
def summary(values):
    return dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),
                std=statistics.stdev(values) if len(values)>1 else 0,min=min(values),max=max(values))
def writecsv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def json_objects(text):
    d=json.JSONDecoder()
    for m in re.finditer(r'(?m)^  \{\s*$',text):
        obj,_=d.raw_decode(text[m.start():].lstrip());yield obj

def ordinary(root,out):
    samples=[];loads=[];paths=[];thermal=[]
    for p in sorted((root/'logs').glob('*.json')):
        if not p.stem.startswith(('standard-','prefix-','pilot-','validate-')):continue
        meta=json.loads(p.read_text())
        if meta['returncode']!=0:continue
        text=p.with_suffix('.txt').read_text(errors='replace');name=p.stem
        conf=re.match(r'(standard|prefix)-(tiny|7b)-(pd0|pd1|layer|row)(?:-b(128|512))?-u(128|512)-g([01])$',name)
        if conf:
            kind,model,split,b,u,glu=conf.groups();common=dict(name=name,kind=kind,model=model,split=split,batch=int(b or u),ubatch=int(u),swiglu=int(glu))
            if kind=='standard':
                for obj in json_objects(text):
                    if 'samples_ns' not in obj:continue
                    phase='pp' if obj['n_prompt'] else 'tg';tokens=obj['n_prompt'] or obj['n_gen']
                    for i,ns in enumerate(obj['samples_ns']):samples.append(dict(**common,context=0 if phase=='tg' else tokens,phase=phase,trial=i,tokens=tokens,ms=ns/1e6,tokens_s=tokens*1e9/ns))
            else:
                for m in re.finditer(r'K200PERF trial=(\d+) pp_ms=([\d.]+) pp_tokens=(\d+) tg_ms=([\d.]+) tg_steps=(\d+) first_process_ms=([\d.]+)',text):
                    trial,pp,context,tg,steps,first=m.groups();context=int(context)
                    for phase,ms,n in [('pp',float(pp),context),('tg',float(tg),int(steps))]:
                        samples.append(dict(**common,context=context,phase=phase,trial=int(trial),tokens=n,ms=ms,tokens_s=n*1000/ms,first_process_ms=float(first)))
                init=re.search(r'K200INIT backend_ms=([\d.]+) model_ms=([\d.]+) context_ms=([\d.]+)',text)
                if init:loads.append(dict(**common,backend_ms=float(init[1]),model_ms=float(init[2]),context_ms=float(init[3])))
                for line in text.splitlines():
                    if line.startswith(('KUNLUN common totals','KUNLUN common phase','KUNLUN SwiGLU totals','KUNLUN Flash totals','KUNLUN Attention totals','KUNLUN common workspace','K200CPU','K200RSS','K200HBM','K200L3')):paths.append(dict(**common,line=line))
        tp=p.parent/(name+'-thermal.jsonl')
        if conf and tp.exists():
            rows=[json.loads(s) for s in tp.read_text().splitlines() if s.strip()]
            for pd in [0,1]:
                dev=[v for sample in rows for v in sample.get('rows',[]) if v['dev']==pd]
                if dev:thermal.append(dict(name=name,pd=pd,max_die_c=max(v['die'] for v in dev),max_hbm_c=max(v['hbm'] for v in dev),min_mhz=min(v['mhz'] for v in dev),max_mhz=max(v['mhz'] for v in dev),sampled_peak_hbm_mib=max(float(v['raw'].split()[17]) for v in dev)))
    groups=defaultdict(list)
    for s in samples:groups[tuple(s[k] for k in ['kind','model','split','batch','ubatch','swiglu','context','phase'])].append(s)
    results=[]
    for key,rows in sorted(groups.items()):
        common=dict(zip(['kind','model','split','batch','ubatch','swiglu','context','phase'],key))
        row=dict(**common)
        for metric in ['tokens_s','ms']:
            for k,v in summary([r[metric] for r in rows]).items():row[metric+'_'+k]=v
        row['tokens_per_trial']=rows[0]['tokens'];results.append(row)
    writecsv(out/'ordinary-samples.csv',samples);writecsv(out/'ordinary-summary.csv',results)
    writecsv(out/'loads.csv',loads);writecsv(out/'dispatch-logs.csv',paths);writecsv(out/'thermal-memory.csv',thermal)
    (out/'ordinary-summary.json').write_text(json.dumps(results,indent=2))
    return results

def read_events(path):
    rows=[]
    with path.open(newline='') as f:
        for r in csv.DictReader(f):
            for k in ['start_ns','end_ns','cpu_ns','bytes','worker','workers','tid']:r[k]=int(r[k])
            assert r['end_ns']>=r['start_ns'],r
            rows.append(r)
    return rows
def interval_index(rows):
    ordered=sorted(rows,key=lambda r:r['start_ns']);starts=[];prefix=[];high=0
    for r in ordered:
        starts.append(r['start_ns']);high=max(high,r['end_ns']);prefix.append(high)
    return ordered,starts,prefix
def enclosing(index,child):
    rows,starts,prefix=index;i=bisect_right(starts,child['start_ns'])-1;best=None
    while i>=0 and prefix[i]>=child['end_ns']:
        r=rows[i]
        if r['end_ns']>=child['end_ns'] and (best is None or r['end_ns']-r['start_ns']<best['end_ns']-best['start_ns']):best=r
        i-=1
    return best
def role(op,name):
    if op=='MUL_MAT':return 'Q4 GEMV/GEMM'
    if op=='FLASH_ATTN_EXT':return 'FlashAttention XPU'
    if op=='ROPE':return 'RoPE XPU'
    if op=='SET_ROWS':return 'KV SET_ROWS XPU'
    if op=='GLU':return 'SwiGLU XPU'
    if op=='RMS_NORM':return 'RMSNorm XPU'
    if op in ['ADD','MUL','SCALE','UNARY']:return 'Elementwise/residual XPU'
    if op in ['CONT','DUP','CPY']:return 'CONT/copy XPU'
    if op in ['VIEW','RESHAPE','PERMUTE','TRANSPOSE','NONE']:return 'XPU metadata'
    return 'Other XPU '+op
def cpu_role(nodes):
    ops={(r['name'],r['tensor']) for r in nodes}
    if any(o=='MUL_MAT' and n=='result_output' for o,n in ops):return 'Q6_K output CPU graph'
    if any(o=='FLASH_ATTN_EXT' for o,n in ops):return 'Attention CPU graph'
    if any(o=='GLU' for o,n in ops):return 'SwiGLU CPU graph'
    if any(o=='ROPE' for o,n in ops):return 'RoPE CPU graph'
    if any(o=='GET_ROWS' and n=='embd' for o,n in ops):return 'Embedding CPU graph'
    if any(o=='GET_ROWS' for o,n in ops):return 'Output selection GET_ROWS CPU graph'
    if all(o in ['VIEW','RESHAPE','PERMUTE','TRANSPOSE','NONE'] for o,n in ops):return 'CPU metadata graph'
    return 'Other CPU graph'

def q4_stages(events):
    """Use existing completion barriers, adding no device synchronization.

    Quantization has 1/2/3 enqueues then a wait; MAC and sum/join have their
    own waits. Thus these are non-overlapping wall stages, including launch,
    driver waiting and overlap of host parameter setup with earlier enqueues.
    FP16 block-scale correction inside GEMM stays fused with its MAC stage.
    """
    ordered=sorted(events,key=lambda r:r['start_ns']);starts=[r['start_ns'] for r in ordered];out=[]
    apis=[r for r in ordered if r['category']=='audit' and r['name'].startswith('q4.common.')]
    def kernel_role(name):
        if 'quant_' in name:return 'Activation quantization'
        if 'ds_dual16' in name:return 'GEMV MAC and device DMA'
        if 'q40_gemm' in name:return 'GEMM MAC, block-scale correction and device DMA'
        if 'q4_sum_partial' in name:return 'GEMV FP32 scale/sum'
        if 'join_slices' in name:return 'GEMM FP32 slice join'
        return 'Other Q4 kernel'
    for api in apis:
        args=None;pending=[];used=[];stage_rows=[]
        for r in ordered[bisect_left(starts,api['start_ns']):bisect_left(starts,api['end_ns'])]:
            if r is api or r['end_ns']>api['end_ns'] or r['category']!='audit':continue
            name=r['name']
            if name.startswith('runtime.arguments.') and args is None:args=r['start_ns']
            elif name.startswith('runtime.launch.'):
                pending.append(r)
                if args is None:args=r['start_ns']
            elif name.startswith('runtime.wait.'):
                if pending:
                    kinds=set(kernel_role(n['name']) for n in pending)
                    role=next(iter(kinds)) if len(kinds)==1 else 'Mixed Q4 stage'
                    assert args<=r['end_ns']
                    stage_rows.append(dict(api=api['name'],stage=role,ns=r['end_ns']-args,kernels=len(pending),calls=1))
                    used.append((args,r['end_ns']));pending=[];args=None
                else:
                    stage_rows.append(dict(api=api['name'],stage='Completed-queue wait',ns=r['end_ns']-r['start_ns'],kernels=0,calls=1));used.append((r['start_ns'],r['end_ns']))
        assert not pending,api
        residual=api['end_ns']-api['start_ns']-length(used);assert residual>=0,api
        stage_rows.append(dict(api=api['name'],stage='Remaining Q4 API host/observer',ns=residual,kernels=0,calls=1))
        for r in stage_rows:r['tensor']=api['tensor']
        out.extend(stage_rows)
    return out

def diagnostics(root,out):
    partitions=[];inclusive=[];cpu=[];copies=[];overlap=[];q4=[];device_copies=[];kv_overhead=[]
    for path in sorted(root.glob('*.events.csv')):
        if path.stem.startswith('pilot'):continue
        name=path.name.removesuffix('.events.csv')
        conf=re.match(r'(?:diag|diagrep)-(tiny|7b)-(pd0|pd1|layer|row)(?:-b(128|512))?-u(128|512)-g([01])$',name)
        if not conf:continue
        print('ANALYZE',path.name,flush=True)
        rows=read_events(path)
        model,split,b,u,glu=conf.groups();common=dict(name=name,model=model,split=split,batch=int(b or u),ubatch=int(u),swiglu=int(glu))
        # Group PP then TG phases. Warmup is absent from timelines; each measured
        # context has a PP period followed by TG, possibly multiple PP processes.
        roots=sorted([r for r in rows if r['category']=='harness' and r['name']=='process'],key=lambda r:r['start_ns'])
        periods=[];last_phase=None
        for r in roots:
            if r['phase']=='pp' and last_phase!='pp':periods.append({'pp':[],'tg':[]})
            periods[-1][r['phase']].append(r);last_phase=r['phase']
        log=(root/'logs'/(name+'.txt')).read_text(errors='replace')
        measured=[(int(t),int(n)) for t,n in re.findall(r'K200PERF trial=(\d+) pp_ms=[\d.]+ pp_tokens=(\d+)',log)]
        assert len(periods)==len(measured),(name,len(periods),len(measured))
        for period_index,((trial,context),period) in enumerate(zip(measured,periods)):
            lo=min(r['start_ns'] for v in period.values() for r in v);hi=max(r['end_ns'] for v in period.values() for r in v)
            if period_index+1<len(periods):hi=min(r['start_ns'] for r in periods[period_index+1]['pp'])-1
            else:hi=max(r['end_ns'] for r in rows)
            selected=[r for r in rows if lo<=r['start_ns'] and r['end_ns']<=hi]
            for phase,phase_roots in period.items():
                rs=[r for r in selected if r['phase']==phase];rootspans=[(r['start_ns'],r['end_ns']) for r in phase_roots]
                total=length(rootspans);units=context if phase=='pp' else len(phase_roots);calls=len(phase_roots)
                c=dict(**common,context=context,trial=trial,phase=phase,process_calls=calls,tokens=units)
                stage_agg=defaultdict(lambda:dict(ns=0,kernels=0,calls=0))
                for r in q4_stages(rs):
                    v=stage_agg[(r['api'],r['stage'],r['tensor'])]
                    for key in ['ns','kernels','calls']:v[key]+=r[key]
                for (api,stage,tensor),v in stage_agg.items():q4.append(dict(**c,api=api,stage=stage,tensor=tensor,calls=v['calls'],kernels=v['kernels'],ms=v['ns']/1e6,ms_per_token=v['ns']/1e6/units))
                candidates=defaultdict(list);counts=defaultdict(int)
                xnodes=[r for r in rs if r['category']=='xpu.node']
                xnode_index=interval_index(xnodes)
                copy_waits=sorted([r for r in rs if r['category']=='audit' and r['name'].startswith('runtime.wait.')],key=lambda r:r['start_ns'])
                copy_wait_starts=[r['start_ns'] for r in copy_waits]
                for r in rs:
                    if r['category']!='audit' or not r['name'].startswith('copy_rows.enqueue.'):continue
                    owner=enclosing(xnode_index,r)
                    candidate=bool(owner and owner['name'] in ['ADD','MUL','RMS_NORM','UNARY','SCALE'])
                    i=bisect_left(copy_wait_starts,r['end_ns']);wait=copy_waits[i] if i<len(copy_waits) else None
                    completion=(wait['end_ns']-r['start_ns'])/1e6 if candidate and wait and wait['end_ns']<=owner['end_ns'] else None
                    device_copies.append(dict(**c,tensor=r['tensor'],op=owner['name'] if owner else '',alias_candidate=candidate,bytes=r['bytes'],enqueue_ms=(r['end_ns']-r['start_ns'])/1e6,completion_ms=completion))
                ordered_rs=sorted(rs,key=lambda r:r['start_ns']);rs_starts=[r['start_ns'] for r in ordered_rs]
                for r in xnodes:
                    k=role(r['name'],r['tensor']);candidates[k].append((r['start_ns'],r['end_ns']));counts[k]+=1
                    if r['name']=='SET_ROWS':
                        inside=[v for v in ordered_rs[bisect_left(rs_starts,r['start_ns']):bisect_left(rs_starts,r['end_ns'])] if v['end_ns']<=r['end_ns']]
                        lib=[(v['start_ns'],v['end_ns']) for v in inside if v['category']=='audit' and v['name'].startswith(('set_rows.api.','kvmax.set_rows.'))]
                        get=[(v['start_ns'],v['end_ns']) for v in inside if v['category'] in ['tensor.get','tensor.get_async']]
                        used=union(lib+get);ns=r['end_ns']-r['start_ns']
                        kv_overhead.append(dict(**c,tensor=r['tensor'],node_ms=ns/1e6,library_union_ms=length(lib)/1e6,index_get_union_ms=length(subtract(get,lib))/1e6,layout_index_validation_host_residual_ms=(ns-length(used))/1e6))
                cnodes=[r for r in rs if r['category']=='cpu.node']
                ordered_cn=sorted(cnodes,key=lambda r:r['start_ns']);cn_starts=[r['start_ns'] for r in ordered_cn]
                cgraphs=[r for r in rs if r['category']=='backend.graph' and r['name']=='CPU']
                for r in cgraphs:
                    contained=[n for n in ordered_cn[bisect_left(cn_starts,r['start_ns']):bisect_left(cn_starts,r['end_ns'])] if n['end_ns']<=r['end_ns']]
                    k=cpu_role(contained);candidates[k].append((r['start_ns'],r['end_ns']));counts[k]+=1
                if not xnodes:
                    # Older pilot has no node scopes. Keep API categories separate.
                    for r in rs:
                        if r['category']=='audit' and r['name'].startswith(('q4.common.','flash.api.','swiglu.api.','elementwise.','attention.rope.api.','kvmax.set_rows.','set_rows.api.')):
                            k=r['name'].split('.')[0]+'.API';candidates[k].append((r['start_ns'],r['end_ns']));counts[k]+=1
                # Actual transfers only. Attention-copy scopes are duplicate views.
                for r in rs:
                    if r['category']=='audit' and r['name'].startswith('runtime.copy.'):
                        k='External '+r['name'].split('.')[2];candidates[k].append((r['start_ns'],r['end_ns']));counts[k]+=1
                for r in rs:
                    if r['category']=='audit' and r['name'].startswith('runtime.wait.'):
                        k='External runtime wait';candidates[k].append((r['start_ns'],r['end_ns']));counts[k]+=1
                for r in rs:
                    if r['category']=='backend.sync':k='External backend synchronization'
                    elif r['category']=='scheduler' and r['name']!='compute_splits':k='Scheduler split/allocation'
                    else:continue
                    candidates[k].append((r['start_ns'],r['end_ns']));counts[k]+=1
                covered=[];partition_total=0
                # Exclusive priority: arithmetic node/CPU graph; then external DMA;
                # then external synchronization; then scheduler preparation.
                for k,spans in candidates.items():
                    clipped=subtract(subtract(spans,subtract(spans,rootspans)),covered)
                    clip_ends=[b for a,b in clipped]
                    active_calls=0
                    for a,b in spans:
                        index=bisect_right(clip_ends,a)
                        if index<len(clipped) and clipped[index][0]<b:active_calls+=1
                    ns=length(clipped);partition_total+=ns;covered=union(covered+clipped)
                    partitions.append(dict(**c,category=k,ms=ns/1e6,ms_per_token=ns/1e6/units,percent=100*ns/total,calls=active_calls,calls_inclusive=counts[k]))
                residual=total-partition_total;assert residual>=0
                partitions.append(dict(**c,category='Remaining host/scheduler/measurement',ms=residual/1e6,ms_per_token=residual/1e6/units,percent=100*residual/total,calls=calls))
                partitions.append(dict(**c,category='TOTAL',ms=total/1e6,ms_per_token=total/1e6/units,percent=100,calls=calls))
                agg=defaultdict(list)
                for r in rs:
                    if r['category']=='audit':agg[(r['category'],r['name'])].append(r)
                    if r['category']=='harness' and r['name'] in ['logits_get','greedy_finite_dump_detokenize']:agg[(r['category'],r['name'])].append(r)
                    if r['category'].startswith('tensor.') and ('result' in r['name'] or 'result' in r['tensor']):agg[(r['category'],r['name'])].append(r)
                for (cat,key),vals in agg.items():
                    inclusive.append(dict(**c,category=cat,key=key,calls=len(vals),ms=sum(r['end_ns']-r['start_ns'] for r in vals)/1e6,bytes=sum(r['bytes'] for r in vals),cpu_ms=sum(r['cpu_ns'] for r in vals)/1e6))
                cg=defaultdict(list)
                for r in cnodes:cg[(r['name'],r['tensor'])].append(r)
                for (op,tensor),vals in cg.items():
                    cpu.append(dict(**c,op=op,tensor=tensor,node_calls=sum(r['worker']==0 for r in vals),worker_calls=len(vals),worker_wall_sum_ms=sum(r['end_ns']-r['start_ns'] for r in vals)/1e6,worker_wall_union_ms=length([(r['start_ns'],r['end_ns']) for r in vals])/1e6,worker_thread_cpu_ms=sum(r['cpu_ns'] for r in vals)/1e6))
                runtime_copies=[r for r in rs if r['category']=='audit' and r['name'].startswith('runtime.copy.')]
                logical_copies=[r for r in rs if r['category'].startswith('tensor.')]
                logical_index=interval_index(logical_copies)
                for r in runtime_copies:
                    owner=enclosing(logical_index,r)
                    copies.append(dict(**c,direction=r['name'].split('.')[2],pd=r['name'].split('.')[-1],bytes=r['bytes'],ms=(r['end_ns']-r['start_ns'])/1e6,tensor=owner['name'] if owner else '(internal API) '+r['tensor'],tensor_dst=owner['tensor'] if owner else ''))
                api=[r for r in rs if r['category']=='audit' and r['name'].startswith(('q4.common.','flash.api.'))]
                waits=[r for r in rs if r['category']=='audit' and r['name'].startswith('runtime.wait.')]
                launches=[r for r in rs if r['category']=='audit' and r['name'].startswith('runtime.launch.')]
                main_tids={r['tid'] for r in phase_roots}
                overlap.append(dict(**c,process_wall_ms=total/1e6,api_union_ms=length([(r['start_ns'],r['end_ns']) for r in api])/1e6,wait_union_ms=length([(r['start_ns'],r['end_ns']) for r in waits])/1e6,launch_union_ms=length([(r['start_ns'],r['end_ns']) for r in launches])/1e6,wait_calls=len(waits),launch_calls=len(launches),main_thread_cpu_ms=sum(r['cpu_ns'] for r in phase_roots)/1e6,other_worker_cpu_ms=sum(r['cpu_ns'] for r in cnodes if r['tid'] not in main_tids)/1e6))
    writecsv(out/'wall-partition.csv',partitions);writecsv(out/'inclusive-api.csv',inclusive);writecsv(out/'cpu-workers.csv',cpu);writecsv(out/'transfers.csv',copies);writecsv(out/'overlap.csv',overlap)
    writecsv(out/'q4-stages.csv',q4)
    writecsv(out/'device-copies.csv',device_copies)
    writecsv(out/'setrows-overhead.csv',kv_overhead)
    return partitions
def tests():
    assert union([(0,10),(5,20),(30,40)])==[(0,20),(30,40)]
    assert subtract([(0,20)],[(2,5),(4,10),(15,30)])==[(0,2),(10,15)]
    assert length([(0,10),(5,20)])==20
    assert subtract([(0,20)],[(0,20)])==[]
    assert summary([1,2,3])['std']==1
def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('root',nargs='?',type=Path,default=Path(__file__).resolve().parent);parser.add_argument('--ordinary-only',action='store_true');args=parser.parse_args()
    tests();root=args.root;out=root/'analysis';out.mkdir(exist_ok=True)
    ordinary(root,out)
    if not args.ordinary_only:diagnostics(root,out)
    print('ANALYSIS_COMPLETE')
if __name__=='__main__':main()

#include <chrono>
#include <cstring>
static void emit_events();
#include "xpu/runtime.h"
#include "k200-profile.h"
#include <dlfcn.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
struct Stat {uint64_t count=0,bytes=0;double ns=0;};
struct State {std::mutex mutex;std::map<std::string,Stat> values;};
static State &state(){static State *s=new State;return *s;}
extern "C" void k200_profile_add(const char *name,double ns,uint64_t bytes){
    const char *phase=std::getenv("K200_PROFILE_PHASE");if(!phase || !std::getenv("K200_ENABLE_PROFILE"))return;
    auto &s=state();std::lock_guard<std::mutex>lock(s.mutex);auto &v=s.values[std::string(phase)+"\t"+name];++v.count;v.bytes+=bytes;v.ns+=ns;
}
extern "C" void k200_profile_emit(){
    auto &s=state();std::lock_guard<std::mutex>lock(s.mutex);
    for(const auto &v:s.values)std::fprintf(stderr,"K200PROFILE\t%s\t%llu\t%.6f\t%llu\n",v.first.c_str(),(unsigned long long)v.second.count,v.second.ns/1e6,(unsigned long long)v.second.bytes);
    s.values.clear();emit_events();
}
static thread_local std::string active_tensor;
static std::atomic<int>pd{-1};
struct Memory {std::mutex mutex;std::map<std::pair<int,void*>,uint64_t> pointers;uint64_t live[2]={},peak[2]={};};
static Memory &l3_memory(){static Memory *m=new Memory;return *m;}
static Memory &memory(){static Memory *m=new Memory;return *m;}
template<class F>static F next(const char *name){auto p=reinterpret_cast<F>(dlsym(RTLD_NEXT,name));if(!p){std::fprintf(stderr,"profiling symbol missing: %s\n",name);std::abort();}return p;}
extern "C" int xpu_set_device(int id){static auto fn=next<int(*)(int)>("xpu_set_device");int rc=fn(id);if(!rc)pd=id;return rc;}
extern "C" int xpu_memcpy(void *dst,const void *src,uint64_t bytes,XPUMemcpyKind kind){
    static auto fn=next<int(*)(void*,const void*,uint64_t,XPUMemcpyKind)>("xpu_memcpy");
    const char *name=kind==XPU_HOST_TO_DEVICE?"H2D":kind==XPU_DEVICE_TO_HOST?"D2H":kind==XPU_DEVICE_TO_DEVICE?"D2D":"other";
    K200Scope scope("runtime.copy."+std::string(name)+".pd"+std::to_string(pd),bytes);
    return fn(dst,src,bytes,kind);
}
extern "C" int xpu_memcpy_peer(int dst,void *dp,int src,const void *sp,uint64_t bytes){
    static auto fn=next<int(*)(int,void*,int,const void*,uint64_t)>("xpu_memcpy_peer");
    K200Scope scope("runtime.copy.P2P."+std::to_string(src)+"to"+std::to_string(dst),bytes);return fn(dst,dp,src,sp,bytes);
}
extern "C" int xpu_wait(XPUStream stream){static auto fn=next<int(*)(XPUStream)>("xpu_wait");K200Scope scope("runtime.wait.inclusive.pd"+std::to_string(pd)+".stream"+std::to_string((uintptr_t)stream));return fn(stream);}

extern "C" int xpu_malloc(void **ptr,uint64_t bytes,XPUMemoryKind kind){
    static auto fn=next<int(*)(void**,uint64_t,XPUMemoryKind)>("xpu_malloc");int rc=fn(ptr,bytes,kind);
    int device=pd.load();
    if(!rc && kind==XPU_MEM_MAIN && device>=0 && device<2){auto &m=memory();std::lock_guard<std::mutex>lock(m.mutex);m.pointers[{device,*ptr}]=bytes;m.live[device]+=bytes;m.peak[device]=std::max(m.peak[device],m.live[device]);}
    if(!rc && kind==XPU_MEM_L3 && device>=0 && device<2){auto &m=l3_memory();std::lock_guard<std::mutex>lock(m.mutex);m.pointers[{device,*ptr}]=bytes;m.live[device]+=bytes;m.peak[device]=std::max(m.peak[device],m.live[device]);}
    return rc;
}
extern "C" int xpu_free(void *ptr){
    static auto fn=next<int(*)(void*)>("xpu_free");int rc=fn(ptr);
    if(!rc){int device=pd.load();auto &m=memory();std::lock_guard<std::mutex>lock(m.mutex);auto it=m.pointers.find({device,ptr});if(it!=m.pointers.end()){m.live[device]-=it->second;m.pointers.erase(it);}}
    if(!rc){int device=pd.load();auto &m=l3_memory();std::lock_guard<std::mutex>lock(m.mutex);auto it=m.pointers.find({device,ptr});if(it!=m.pointers.end()){m.live[device]-=it->second;m.pointers.erase(it);}}
    return rc;
}
extern "C" void k200_profile_memory(){
 {auto&m=l3_memory();std::lock_guard<std::mutex>lock(m.mutex);std::fprintf(stderr,"K200L3 api_peak_pd0=%llu api_peak_pd1=%llu api_live_pd0=%llu api_live_pd1=%llu\n",(unsigned long long)m.peak[0],(unsigned long long)m.peak[1],(unsigned long long)m.live[0],(unsigned long long)m.live[1]);}
 auto&m=memory();std::lock_guard<std::mutex>lock(m.mutex);std::fprintf(stderr,"K200HBM api_peak_pd0=%llu api_peak_pd1=%llu api_live_pd0=%llu api_live_pd1=%llu\n",(unsigned long long)m.peak[0],(unsigned long long)m.peak[1],(unsigned long long)m.live[0],(unsigned long long)m.live[1]);}
#include "ggml.h"
#include <kunlun/k200.h>
namespace kunlun {
bool set_rows_f16(Kernel &k,int id,void *d,size_t db,const void *x,size_t xb,const void *i,size_t ib,const SetRowsLayout &p) {
    using Fn=bool(*)(Kernel&,int,void*,size_t,const void*,size_t,const void*,size_t,const SetRowsLayout&);
    static auto fn=next<Fn>("_ZN6kunlun12set_rows_f16ERNS_6KernelEiPvmPKvmS4_mRKNS_13SetRowsLayoutE");
    K200Scope scope("set_rows.api.pd"+std::to_string(id)+".width"+std::to_string(p.width)+".tokens"+std::to_string(p.cache_tokens),xb+ib);
    return fn(k,id,d,db,x,xb,i,ib,p);
}
}

namespace kunlun {
bool swiglu(Kernel &k,int id,void *y,size_t yb,const void *g,size_t gb,const void *u,size_t ub,const SwiGLULayout &p) {
    using Fn=bool(*)(Kernel&,int,void*,size_t,const void*,size_t,const void*,size_t,const SwiGLULayout&);
    static auto fn=next<Fn>("_ZN6kunlun6swigluERNS_6KernelEiPvmPKvmS4_mRKNS_12SwiGLULayoutE");
    K200Scope scope("swiglu.api.pd"+std::to_string(id)+".w"+std::to_string(p.width)+".m"+std::to_string(p.rows));
    return fn(k,id,y,yb,g,gb,u,ub,p);
}
}

namespace kunlun {
bool attention_matmul(Kernel &k,int id,void *y,size_t yb,const void *a,size_t ab,const void *b,size_t bb,const Matmul4Layout &p) {
 using Fn=bool(*)(Kernel&,int,void*,size_t,const void*,size_t,const void*,size_t,const Matmul4Layout&);
 static auto fn=next<Fn>("_ZN6kunlun16attention_matmulERNS_6KernelEiPvmPKvmS4_mRKNS_13Matmul4LayoutE");K200Scope scope("attention.matmul.api.pd"+std::to_string(id));
 return fn(k,id,y,yb,a,ab,b,bb,p);
}
}

namespace kunlun {
bool attention_softmax(Kernel &k,int id,void *y,size_t yb,const void *a,size_t ab,const void *b,size_t bb,const Softmax4Layout &p) {
 using Fn=bool(*)(Kernel&,int,void*,size_t,const void*,size_t,const void*,size_t,const Softmax4Layout&);
 static auto fn=next<Fn>("_ZN6kunlun17attention_softmaxERNS_6KernelEiPvmPKvmS4_mRKNS_14Softmax4LayoutE");K200Scope scope("attention.softmax.api.pd"+std::to_string(id));
 return fn(k,id,y,yb,a,ab,b,bb,p);
}
}

namespace kunlun {
bool attention_contiguous(Kernel &k,int id,void *y,size_t yb,const void *a,size_t ab,const Tensor4Layout &p) {
 using Fn=bool(*)(Kernel&,int,void*,size_t,const void*,size_t,const Tensor4Layout&);
 static auto fn=next<Fn>("_ZN6kunlun20attention_contiguousERNS_6KernelEiPvmPKvmRKNS_13Tensor4LayoutE");K200Scope scope("attention.cont.api.pd"+std::to_string(id));return fn(k,id,y,yb,a,ab,p);
}
bool attention_rope(Kernel &k,int id,void *y,size_t yb,const void *a,size_t ab,const int32_t *pos,size_t pb,const Rope4Layout &p) {
 using Fn=bool(*)(Kernel&,int,void*,size_t,const void*,size_t,const int32_t*,size_t,const Rope4Layout&);
 static auto fn=next<Fn>("_ZN6kunlun14attention_ropeERNS_6KernelEiPvmPKvmPKimRKNS_11Rope4LayoutE");K200Scope scope("attention.rope.api.pd"+std::to_string(id));return fn(k,id,y,yb,a,ab,pos,pb,p);
}
}

#include "k200-diag.h"
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>
#include <cinttypes>
struct Event {std::string phase,category,name,tensor;uint64_t start,end,cpu,bytes;int worker,workers;long tid;};
static std::vector<Event>&events(){static auto *v=new std::vector<Event>;return *v;}
static std::mutex&event_mutex(){static auto*m=new std::mutex;return *m;}
extern "C" void k200_diag_event(const char*c,const char*n,const char*t,int w,int nw,uint64_t s,uint64_t e,uint64_t cpu,uint64_t bytes){
 const char*p=getenv("K200_PROFILE_PHASE");if(!p||!getenv("K200_ENABLE_PROFILE"))return;
 // Exclude loading/warmup from the full timeline, retain aggregate memory/counters.
 if(strcmp(p,"pp")&&strcmp(p,"tg"))return;
 Event v{p,c,n,(!strcmp(c,"audit") && (!t || !*t))?active_tensor:(t?t:""),s,e,cpu,bytes,w,nw,(long)syscall(SYS_gettid)};
 std::lock_guard<std::mutex>lock(event_mutex());events().push_back(std::move(v));
}
static std::string csv_cell(const std::string &v){std::string r(1,char(34));for(char c:v){r.push_back(c);if(c==34)r.push_back(c);}r.push_back(34);return r;}
static void emit_events(){
 const char*p=getenv("K200_TRACE_PATH");if(!p)return;FILE*f=fopen(p,"w");if(!f){perror(p);abort();}
 fprintf(f,"phase,category,name,tensor,start_ns,end_ns,cpu_ns,bytes,worker,workers,tid\n");
 std::lock_guard<std::mutex>lock(event_mutex());
 for(auto&e:events())fprintf(f,"%s,%s,%s,%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d,%d,%ld\n",csv_cell(e.phase).c_str(),csv_cell(e.category).c_str(),csv_cell(e.name).c_str(),csv_cell(e.tensor).c_str(),e.start,e.end,e.cpu,e.bytes,e.worker,e.workers,e.tid);
 fclose(f);events().clear();
}
static std::map<uintptr_t,std::string>&programs(){static auto*p=new std::map<uintptr_t,std::string>;return *p;}
extern "C" int xpu_create_sd_func(XPUFunc*f,uint64_t a,uint32_t b,uint32_t pc,uint32_t ps,uint64_t hash,const char*n,bool on){
 static auto fn=next<int(*)(XPUFunc*,uint64_t,uint32_t,uint32_t,uint32_t,uint64_t,const char*,bool)>("xpu_create_sd_func");int rc=fn(f,a,b,pc,ps,hash,n,on);if(!rc)programs()[(uintptr_t)*f]=n?n:"sd";return rc;
}
extern "C" int xpu_create_cl_func(XPUFunc*f,uint64_t a,uint32_t b,uint32_t pc,uint32_t ps,uint64_t hash,const char*n,bool on){
 static auto fn=next<int(*)(XPUFunc*,uint64_t,uint32_t,uint32_t,uint32_t,uint64_t,const char*,bool)>("xpu_create_cl_func");int rc=fn(f,a,b,pc,ps,hash,n,on);if(!rc)programs()[(uintptr_t)*f]=n?n:"cl";return rc;
}
static thread_local XPUStream configured_stream=nullptr;
static thread_local uint64_t argument_start=0,argument_bytes=0;
extern "C" int xpu_launch_argument_set(const void *arg,size_t bytes,size_t offset){
 static auto fn=next<int(*)(const void*,size_t,size_t)>("xpu_launch_argument_set");
 if(!offset){argument_start=k200_diag_clock(CLOCK_MONOTONIC);argument_bytes=0;}
 argument_bytes+=bytes;return fn(arg,bytes,offset);
}

extern "C" int xpu_launch_config(int clusters,int cores,XPUStream stream){static auto fn=next<int(*)(int,int,XPUStream)>("xpu_launch_config");configured_stream=stream;
 if(argument_start){auto end=k200_diag_clock(CLOCK_MONOTONIC);std::string name="runtime.arguments.pd"+std::to_string(pd);k200_profile_add(name.c_str(),end-argument_start,argument_bytes);k200_diag_event("audit",name.c_str(),"",-1,1,argument_start,end,0,argument_bytes);argument_start=0;}
 K200Scope scope("runtime.launch_config.pd"+std::to_string(pd)+".stream"+std::to_string((uintptr_t)stream));return fn(clusters,cores,stream);}
extern "C" int xpu_launch_async(XPUFunc func){
 static auto fn=next<int(*)(XPUFunc)>("xpu_launch_async");auto it=programs().find((uintptr_t)func);std::string symbol=it==programs().end()?"SDK.unmapped":it->second;
 int rc;{K200Scope scope("runtime.launch."+symbol+".pd"+std::to_string(pd)+".stream"+std::to_string((uintptr_t)configured_stream));rc=fn(func);}
 if(!rc&&getenv("K200_SERIAL_KERNELS")){static auto wait_fn=next<int(*)(XPUStream)>("xpu_wait");K200Scope scope("kernel.serial_wait."+symbol+".pd"+std::to_string(pd)+".stream"+std::to_string((uintptr_t)configured_stream));rc=wait_fn(configured_stream);}
 return rc;
}
namespace kunlun {
void diagnostic_tensor(const char*n,const int64_t*d){using F=void(*)(const char*,const int64_t*);static auto fn=next<F>("_ZN6kunlun17diagnostic_tensorEPKcPKl");active_tensor=n?n:"";fn(n,d);}
void common_q4_matmul(CommonQ4Workspace&w,int id,void*y,const CommonQ4Weight&weight,const void*x,int m,InputType type){
 using F=void(*)(CommonQ4Workspace&,int,void*,const CommonQ4Weight&,const void*,int,InputType);static auto fn=next<F>("_ZN6kunlun16common_q4_matmulERNS_17CommonQ4WorkspaceEiPvRKNS_14CommonQ4WeightEPKviNS_9InputTypeE");auto i=weight.info();
 K200Scope scope("q4.common."+std::string(m==1?"gemv":"gemm")+".pd"+std::to_string(id)+".k"+std::to_string(i.k)+".n"+std::to_string(i.n)+".m"+std::to_string(m));fn(w,id,y,weight,x,m,type);
}
bool flash_attention(Kernel&k,int id,void*y,size_t yb,const void*q,size_t qb,const void*key,size_t kb,const void*v,size_t vb,const void*mask,size_t mb,const Flash4Layout&p){
 using F=bool(*)(Kernel&,int,void*,size_t,const void*,size_t,const void*,size_t,const void*,size_t,const void*,size_t,const Flash4Layout&);static auto fn=next<F>("_ZN6kunlun15flash_attentionERNS_6KernelEiPvmPKvmS4_mS4_mS4_mRKNS_12Flash4LayoutE");K200Scope scope("flash.api.pd"+std::to_string(id)+".d"+std::to_string(p.q.ne[0])+".m"+std::to_string(p.q.ne[1])+".s"+std::to_string(p.k.ne[1]));return fn(k,id,y,yb,q,qb,key,kb,v,vb,mask,mb,p);
}
void elementwise(Kernel&k,int id,void*y,const void*a,const void*b,int count,int width,int bcount,ElementwiseOp op,float parameter){
 using F=void(*)(Kernel&,int,void*,const void*,const void*,int,int,int,ElementwiseOp,float);static auto fn=next<F>("_ZN6kunlun11elementwiseERNS_6KernelEiPvPKvS4_iiiNS_13ElementwiseOpEf");K200Scope scope("elementwise.op"+std::to_string((int)op)+".pd"+std::to_string(id));fn(k,id,y,a,b,count,width,bcount,op,parameter);
}
void rms_norm(Kernel&k,int id,void*y,const void*x,int width,int rows,float eps){
 using F=void(*)(Kernel&,int,void*,const void*,int,int,float);static auto fn=next<F>("_ZN6kunlun8rms_normERNS_6KernelEiPvPKviif");K200Scope scope("rms_norm.api.pd"+std::to_string(id));fn(k,id,y,x,width,rows,eps);
}
bool kv_max_set_rows(Kernel&k,int id,void*d,size_t db,const void*x,size_t xb,const void*i,size_t ib,const SetRowsLayout&p,const int64_t*t,size_t n){
 using F=bool(*)(Kernel&,int,void*,size_t,const void*,size_t,const void*,size_t,const SetRowsLayout&,const int64_t*,size_t);static auto fn=next<F>("_ZN6kunlun15kv_max_set_rowsERNS_6KernelEiPvmPKvmS4_mRKNS_13SetRowsLayoutEPKlm");K200Scope scope("kvmax.set_rows.pd"+std::to_string(id));return fn(k,id,d,db,x,xb,i,ib,p,t,n);
}
}

namespace kunlun {
void peer(int dst,void *dp,int src,const void *sp,size_t bytes){
 using F=void(*)(int,void*,int,const void*,size_t);
 static auto fn=next<F>("_ZN6kunlun4peerEiPviPKvm");
 K200Scope scope("peer.ready_inclusive."+std::to_string(src)+"to"+std::to_string(dst),bytes);
 fn(dst,dp,src,sp,bytes);
}
void copy_rows(Kernel &k,int id,void *dst,const void *src,int width,int rows,int stride,int column){
 using F=void(*)(Kernel&,int,void*,const void*,int,int,int,int);
 static auto fn=next<F>("_ZN6kunlun9copy_rowsERNS_6KernelEiPvPKviiii");
 K200Scope scope("copy_rows.enqueue.pd"+std::to_string(id)+".width"+std::to_string(width)+".rows"+std::to_string(rows),(uint64_t)width*rows*4);
 fn(k,id,dst,src,width,rows,stride,column);
}
}

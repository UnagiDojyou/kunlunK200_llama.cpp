// Low-overhead CPU-output-head observation; no XPU runtime interposition.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mutex>
#include <cstdint>
#include <unistd.h>
#include <sys/syscall.h>
struct Event {std::string phase,category,name,tensor;int worker,workers;uint64_t begin,end,cpu,bytes;long tid;};
static std::vector<Event>&events(){static auto*p=new std::vector<Event>;return *p;}
static std::mutex&mutex(){static auto*p=new std::mutex;return *p;}
static std::string csv_cell(const std::string &v){std::string r(1,char(34));for(char c:v){r.push_back(c);if(c==34)r.push_back(c);}r.push_back(34);return r;}
extern "C" void k200_diag_event(const char*c,const char*n,const char*t,int w,int nw,uint64_t b,uint64_t e,uint64_t cpu,uint64_t bytes){


 const char*p=getenv("K200_PROFILE_PHASE");if(!p||(strcmp(p,"pp")&&strcmp(p,"tg")))return;
 Event v{p,c,n,t,w,nw,b,e,cpu,bytes,(long)syscall(SYS_gettid)};
 std::lock_guard<std::mutex>lock(mutex());events().push_back(std::move(v));
}
extern "C" void k200_profile_emit(){
 const char*p=getenv("K200_TRACE_PATH");if(!p)return;FILE*f=fopen(p,"w");if(!f)abort();
 fprintf(f,"phase,category,name,tensor,start_ns,end_ns,cpu_ns,bytes,worker,workers,tid\n");
 for(auto&r:events())fprintf(f,"%s,%s,%s,%s,%llu,%llu,%llu,%llu,%d,%d,%ld\n",csv_cell(r.phase).c_str(),csv_cell(r.category).c_str(),csv_cell(r.name).c_str(),csv_cell(r.tensor).c_str(),(unsigned long long)r.begin,(unsigned long long)r.end,(unsigned long long)r.cpu,(unsigned long long)r.bytes,r.worker,r.workers,r.tid);
 fclose(f);events().clear();
}

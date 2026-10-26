#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <sys/resource.h>

using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
static int setting(const char *name,int fallback) {
    const char *value=std::getenv(name);
    if(!value)return fallback;
    char *end=nullptr;long result=std::strtol(value,&end,10);
    if(end==value || *end || result<1 || result>1048576){std::fprintf(stderr,"invalid %s\n",name);std::exit(2);}
    return result;
}
static std::vector<llama_token> read_ids(const char *file,const llama_vocab *vocab) {
    std::ifstream input(file);std::vector<llama_token> ids;long id;
    if(!input){std::fprintf(stderr,"cannot read %s\n",file);std::exit(2);}
    while(input>>id){if(id<0 || id>=llama_vocab_n_tokens(vocab))std::exit(2);ids.push_back(id);}
    if(!input.eof())std::exit(2);
    return ids;
}
// Diagnostic only: scheduler callbacks synchronize each observed node.
// Never use these perturbed times as the primary end-to-end benchmark.
struct OpTrace {
    struct Value { size_t calls=0; double ms=0; };
    Clock::time_point start;
    std::map<std::string,Value> values;
};
static bool trace_op(ggml_tensor * node,bool ask,void * opaque) {
    auto & trace=*static_cast<OpTrace *>(opaque);
    if(ask) {
        if(std::getenv("K200_TRACE_LAYOUT") && (node->op==GGML_OP_MUL_MAT || node->op==GGML_OP_SOFT_MAX || node->op==GGML_OP_ROPE || node->op==GGML_OP_CONT)) {
            static std::map<std::string,bool> seen;
            std::string key=std::string(node->name)+":"+std::to_string(node->ne[1]);
            if(!seen[key] && (std::strstr(node->name,"-0") || node->op==GGML_OP_SOFT_MAX)) {
                seen[key]=true;
                auto log=[](const char *role,const ggml_tensor *t) {
                    if(!t)return;
                    std::fprintf(stderr,"K200LAYOUT role=%s name=%s op=%s type=%s ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu buffer=%s params=",role,t->name,ggml_op_name(t->op),ggml_type_name(t->type),(long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3],t->nb[0],t->nb[1],t->nb[2],t->nb[3],t->buffer?ggml_backend_buffer_name(t->buffer):"view");
                    for(int i=0;i<8;++i)std::fprintf(stderr,"%s%d",i?",":"",t->op_params[i]);std::fprintf(stderr,"\n");
                };
                log("dst",node);for(const auto *v:node->src)log("src",v);
            }
        }
        trace.start=Clock::now();return true;
    }
    double ms=elapsed(trace.start);
    const char *phase=std::getenv("K200_PROFILE_PHASE");
    std::string key=std::string(phase?phase:"other")+"|"+ggml_op_name(node->op)+"|"+
        (node->buffer?ggml_backend_buffer_name(node->buffer):"view")+"|"+
        (node->src[0]?ggml_type_name(node->src[0]->type):"none")+"|"+
        std::to_string(node->ne[0])+"x"+std::to_string(node->ne[1]);
    if(const char *dir=std::getenv("K200_CAPTURE_GLU")) {
        int layer=std::getenv("K200_CAPTURE_LAYER")?std::atoi(std::getenv("K200_CAPTURE_LAYER")):0;
        bool chosen=false;
        for(const char *prefix:{"ffn_norm","ffn_gate","ffn_up","ffn_swiglu","ffn_down","ffn_out","l_out"})
            chosen=chosen || (layer<0?std::string(node->name).find(std::string(prefix)+"-")==0:std::string(node->name)==std::string(prefix)+"-"+std::to_string(layer));
        if(chosen && phase && !std::strcmp(phase,"pp") && node->type==GGML_TYPE_F32) {
            std::vector<char>raw(ggml_nbytes(node));ggml_backend_tensor_get(node,raw.data(),0,raw.size());
            std::string file=std::string(dir)+"/"+node->name+".bin";
            FILE *f=std::fopen(file.c_str(),"wb");if(!f || std::fwrite(raw.data(),1,raw.size(),f)!=raw.size())std::exit(2);std::fclose(f);
            std::fprintf(stderr,"K200CAP name=%s bytes=%zu shape=%lldx%lldx%lldx%lld\n",node->name,raw.size(),(long long)node->ne[0],(long long)node->ne[1],(long long)node->ne[2],(long long)node->ne[3]);
        }
    }
    auto & v=trace.values[key];++v.calls;v.ms+=ms;return true;
}
static int smoke_run(int argc,char **argv) {
    if(argc<3){std::fprintf(stderr,"usage: %s model.gguf cpu|none|layer|row [positions] [logits.bin] [device]\n",argv[0]);return 2;}
    const char *mode=argv[2];bool cpu=std::strcmp(mode,"cpu")==0;
    if(!cpu && std::strcmp(mode,"none") && std::strcmp(mode,"layer") && std::strcmp(mode,"row"))return 2;
    int predict=argc>3?std::atoi(argv[3]):8,device=argc>5?std::atoi(argv[5]):0;
    if(predict<1 || predict>8192 || device<0 || device>1 || (std::strcmp(mode,"none") && device))return 2;
    auto start=Clock::now();llama_backend_init();double backend_ms=elapsed(start);
    ggml_backend_dev_t devices[3]={nullptr,nullptr,nullptr};
    if(!cpu){
        devices[0]=ggml_backend_dev_by_name(device?"KUNLUN1":"KUNLUN0");if(!devices[0])return 1;
        if(std::strcmp(mode,"none") && !(std::getenv("K200_ROW_MAIN_ONLY") && !std::strcmp(mode,"row"))){
            devices[1]=ggml_backend_dev_by_name("KUNLUN1");if(!devices[1])return 1;
        }
    }
    float split[2]={1,1};auto mp=llama_model_default_params();
    mp.devices=devices;mp.n_gpu_layers=cpu?0:99;mp.main_gpu=0;mp.tensor_split=split;
    mp.split_mode=!std::strcmp(mode,"row")?LLAMA_SPLIT_MODE_ROW:!std::strcmp(mode,"layer")?LLAMA_SPLIT_MODE_LAYER:LLAMA_SPLIT_MODE_NONE;
    setenv("K200_PROFILE_PHASE","model",1);
    start=Clock::now();auto *model=llama_model_load_from_file(argv[1],mp);if(!model)return 1;
    double model_ms=elapsed(start);auto cp=llama_context_default_params();
    cp.n_ctx=setting("K200_CTX",256);cp.n_batch=setting("K200_BATCH",8);cp.n_ubatch=setting("K200_UBATCH",8);
    cp.n_threads=6;cp.n_threads_batch=6;cp.offload_kqv=std::getenv("GGML_KUNLUN_CPU_KV")==nullptr;
    cp.flash_attn_type=(std::getenv("K200_FLASH")&&!std::strcmp(std::getenv("K200_FLASH"),"1"))?LLAMA_FLASH_ATTN_TYPE_ENABLED:LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.no_perf=false;
    OpTrace trace;
    if(std::getenv("K200_TRACE_OPS")) { cp.cb_eval=trace_op;cp.cb_eval_user_data=&trace; }
    setenv("K200_PROFILE_PHASE","context",1);
    start=Clock::now();auto *ctx=llama_init_from_model(model,cp);if(!ctx)return 1;
    double context_ms=elapsed(start);auto *vocab=llama_model_get_vocab(model);
    std::vector<llama_token>teacher,tokens;
    if(const char *file=std::getenv("K200_TEACHER_FILE"))teacher=read_ids(file,vocab);
    else if(const char *ids=std::getenv("GGML_KUNLUN_TEACHER_TOKENS")){
        while(*ids){if(*ids==' ' || *ids==','){++ids;continue;}char *end=nullptr;long id=std::strtol(ids,&end,10);
            if(end==ids || id<0 || id>=llama_vocab_n_tokens(vocab))return 2;
            teacher.push_back(id);ids=end;
        }
    }
    if(!teacher.empty() && (int)teacher.size()<predict)return 2;
    std::string prompt="The capital of France is";
    if(const char *file=std::getenv("K200_PROMPT_FILE")){
        std::ifstream in(file);if(!in)return 2;prompt.assign(std::istreambuf_iterator<char>(in),{});
    }
    if(const char *file=std::getenv("K200_INPUT_IDS"))tokens=read_ids(file,vocab);
    else {
        int nt=-llama_tokenize(vocab,prompt.data(),prompt.size(),nullptr,0,true,true);if(nt<1)return 2;
        tokens.resize(nt);if(llama_tokenize(vocab,prompt.data(),prompt.size(),tokens.data(),nt,true,true)!=nt)return 2;
    }
    if(tokens.empty() || tokens.size()+predict-1>cp.n_ctx)return 2;
    std::fprintf(stderr,"K200INPUT");for(auto id:tokens)std::fprintf(stderr," %d",id);std::fprintf(stderr,"\n");
    std::fprintf(stderr,"K200INIT backend_ms=%.3f model_ms=%.3f context_ms=%.3f ctx=%u batch=%u ubatch=%u prompt_tokens=%zu positions=%d teacher_forced=%d\n",backend_ms,model_ms,context_ms,cp.n_ctx,cp.n_batch,cp.n_ubatch,tokens.size(),predict,!teacher.empty() || std::getenv("K200_TEACHER_DIR"));
    auto *batch=llama_batch_ext_init(ctx);
    auto process=[&](const llama_token *ids,int count,int pos){
        llama_batch_ext_clear(batch);
        for(int i=0;i<count;++i){int j=llama_batch_ext_add_token(batch,0,ids[i]);llama_pos p=pos+i;llama_batch_ext_set_pos(batch,j,&p);}
        llama_batch_ext_set_output_logits(batch,count-1,true);
        if(llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,batch))return (float *)nullptr;
        return llama_get_logits_ith(ctx,-1);
    };

    // Controlled A/B benchmark only. Save the common prefill once; restoring
    // the identical KV state is outside every timed TG region. No trace callback
    // or additional wait is used inside process(). Rotate modes between trials.
    if(teacher.empty())return 2;
    const auto all_tokens=tokens;std::vector<size_t> lengths;
    if(const char *raw=std::getenv("K200_PP_LENGTHS"))while(*raw){
        if(*raw==','||*raw==' '){++raw;continue;}char *end=nullptr;long n=std::strtol(raw,&end,10);
        if(end==raw||n<1||(size_t)n>all_tokens.size())return 2;lengths.push_back(n);raw=end;
    }else lengths.push_back(all_tokens.size());
    std::vector<std::string> modes;std::string raw=std::getenv("K200_FD_MODES")?std::getenv("K200_FD_MODES"):"off,on";
    for(size_t first=0;first<raw.size();){size_t end=raw.find(',',first);if(end==std::string::npos)end=raw.size();modes.push_back(raw.substr(first,end-first));first=end+1;}
    int repeats=setting("K200_REPEATS",3),nv=llama_vocab_n_tokens(vocab);
    for(size_t length:lengths){
        tokens.assign(all_tokens.begin(),all_tokens.begin()+length);llama_memory_clear(llama_get_memory(ctx),true);setenv("KUNLUN_FLASH_DECODING","off",1);
        float *logits=nullptr;double pp_ms=0;
        for(size_t pos=0;pos<tokens.size();){int count=std::min<size_t>(cp.n_batch,tokens.size()-pos);start=Clock::now();logits=process(tokens.data()+pos,count,pos);pp_ms+=elapsed(start);if(!logits)return 1;pos+=count;}
        std::vector<float> initial(logits,logits+nv);size_t bytes=llama_state_get_size(ctx);if(!bytes||bytes>size_t(4)*1024*1024*1024)return 1;
        std::vector<uint8_t> state(bytes);size_t saved=llama_state_get_data(ctx,state.data(),state.size());if(!saved||saved>state.size())return 1;state.resize(saved);
        std::fprintf(stderr,"FDBENCH_STATE prefix=%zu bytes=%zu prefill_ms=%.6f restore_outside_timer=1\n",length,saved,pp_ms);
        std::map<std::string,unsigned long long> checksums;
        for(int trial=-1;trial<repeats;++trial)for(size_t order=0;order<modes.size();++order){
            const auto &md=modes[(order+(trial+1))%modes.size()];std::string mode=md;size_t colon=md.find(':');
            unsetenv("KUNLUN_FLASH_DECODING_CHUNKS");
            if(colon!=std::string::npos){mode=md.substr(0,colon);setenv("KUNLUN_FLASH_DECODING_CHUNKS",md.substr(colon+1).c_str(),1);}
            unsetenv("KUNLUN_FLASH_DECODING_FIXED_KEYS");if(mode=="fixed"||mode=="fixhead"){setenv("KUNLUN_FLASH_DECODING_FIXED_KEYS","1024",1);mode=mode=="fixhead"?"head":"on";}unsetenv("KUNLUN_FLASH_DECODING_WORK");if(mode=="head"){setenv("KUNLUN_FLASH_DECODING_WORK","head",1);mode="on";}if(mode!="on"&&mode!="off"&&mode!="auto")return 2;setenv("KUNLUN_FLASH_DECODING",mode.c_str(),1);
            llama_memory_clear(llama_get_memory(ctx),true); // Match fresh prefill padding in every mode.
            if(llama_state_set_data(ctx,state.data(),state.size())!=state.size())return 1;logits=initial.data();
            FILE *dump=nullptr;if(trial==0&&argc>4){std::string file=std::string(argv[4])+".p"+std::to_string(length)+"."+md;dump=std::fopen(file.c_str(),"wb");if(!dump)return 1;}
            double tg_ms=0;int steps=0;unsigned long long hash=1469598103934665603ull;
            for(int i=0;i<predict;++i){
                for(int j=0;j<nv;++j){if(!std::isfinite(logits[j]))return 1;unsigned int word;std::memcpy(&word,logits+j,4);hash^=word;hash*=1099511628211ull;}
                if(dump&&std::fwrite(logits,4,nv,dump)!=(size_t)nv)return 1;
                if(i+1==predict)break;llama_token next=teacher[i];start=Clock::now();logits=process(&next,1,tokens.size()+i);tg_ms+=elapsed(start);++steps;if(!logits)return 1;
            }
            if(dump)std::fclose(dump);
            if(!checksums.count(md))checksums[md]=hash;else if(checksums[md]!=hash){std::fprintf(stderr,"non-repeatable logits mode=%s\n",md.c_str());return 1;}
            if(trial>=0)std::fprintf(stderr,"FDBENCH trial=%d order=%zu prefix=%zu mode=%s tg_ms=%.6f steps=%d tps=%.9f logits_hash=%llu\n",trial,order,length,md.c_str(),tg_ms,steps,steps*1000./tg_ms,hash);
        }
    }
    llama_batch_ext_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}
int main(int argc,char **argv){return smoke_run(argc,argv);}

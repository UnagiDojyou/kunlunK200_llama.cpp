#include "../../ggml/src/k200-diag.h"
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
        if(std::getenv("K200_TRACE_LAYOUT") && (node->op==GGML_OP_MUL_MAT || node->op==GGML_OP_SOFT_MAX || node->op==GGML_OP_ROPE || node->op==GGML_OP_CONT || node->op==GGML_OP_GET_ROWS || node->op==GGML_OP_SET_ROWS || node->op==GGML_OP_GLU)) {
            static std::map<std::string,bool> seen;
            std::string key=std::string(node->name)+":"+std::to_string(node->ne[1]);
            if(!seen[key] && (std::strstr(node->name,"-0") || node->op==GGML_OP_SOFT_MAX || node->op==GGML_OP_GET_ROWS || !std::strcmp(node->name,"result_output"))) {
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
    struct Prompt {std::string id,path;};
    std::vector<Prompt> prompts{{"single",""}};
    bool task_set=std::getenv("K200_TASK_MANIFEST")!=nullptr;
    if(task_set){
        if(std::getenv("K200_INPUT_IDS") || std::getenv("K200_PP_LENGTHS") || !teacher.empty())return 2;
        prompts.clear();std::ifstream input(std::getenv("K200_TASK_MANIFEST"));std::string id,path;
        while(input>>id>>path){if(id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos)return 2;prompts.push_back({id,path});}
        if(!input.eof() || prompts.empty())return 2;
    }
    auto *batch=llama_batch_ext_init(ctx);
    for(const auto &task:prompts) {
    if(task_set)std::fprintf(stderr,"K200CASE_BEGIN id=%s\n",task.id.c_str());
    std::string prompt="The capital of France is";
    if(const char *file=task.path.empty()?std::getenv("K200_PROMPT_FILE"):task.path.c_str()){
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
    auto process=[&](const llama_token *ids,int count,int pos){
        K200_DIAG_SCOPE(process_scope,"harness","process");
        llama_batch_ext_clear(batch);
        for(int i=0;i<count;++i){int j=llama_batch_ext_add_token(batch,0,ids[i]);llama_pos p=pos+i;llama_batch_ext_set_pos(batch,j,&p);}
        llama_batch_ext_set_output_logits(batch,count-1,true);
        { K200_DIAG_SCOPE(scope,"harness","llama_process");if(llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,batch))return (float *)nullptr; }
        K200_DIAG_SCOPE(scope,"harness","logits_get");return llama_get_logits_ith(ctx,-1);
    };
    int repeats=setting("K200_REPEATS",1),warmup=std::getenv("K200_WARMUP") && std::strcmp(std::getenv("K200_WARMUP"),"0")!=0;
    const auto all_tokens=tokens;
    std::vector<size_t> lengths;
    if(const char * raw=std::getenv("K200_PP_LENGTHS")) {
        while(*raw){if(*raw==' ' || *raw==','){++raw;continue;}char *end=nullptr;long v=std::strtol(raw,&end,10);
            if(end==raw || v<1 || (size_t)v>all_tokens.size())return 2;
            lengths.push_back(v);raw=end;
        }
    } else lengths.push_back(all_tokens.size());
    for(size_t length:lengths) {
    tokens.assign(all_tokens.begin(),all_tokens.begin()+length);
    if(const char *dir=std::getenv("K200_TEACHER_DIR")) {
        std::string file=std::string(dir)+"/p"+std::to_string(length)+".ids";
        teacher=read_ids(file.c_str(),vocab);if((int)teacher.size()<predict)return 2;
    }
    for(int trial=-warmup;trial<repeats;++trial){
        setenv("K200_PROFILE_PHASE","clear",1);
        llama_memory_clear(llama_get_memory(ctx),true);
        llama_perf_context_reset(ctx);
        rusage cpu_start{};getrusage(RUSAGE_SELF,&cpu_start);
        float *logits=nullptr;double pp_ms=0,tg_ms=0,first_ms=0;int decode_steps=0;
        setenv("K200_PROFILE_PHASE",trial<0?"warmup.pp":"pp",1);
        for(size_t pos=0;pos<tokens.size();){
            int count=std::min<size_t>(cp.n_batch,tokens.size()-pos);
            start=Clock::now();logits=process(tokens.data()+pos,count,pos);double ms=elapsed(start);
            if(!logits)return 1;
            if(!pos)first_ms=ms;
            pp_ms+=ms;pos+=count;
        }
        std::vector<float> state_logits;
        if(std::getenv("K200_CHECK_STATE")) {
            setenv("K200_PROFILE_PHASE","state_check",1);
            int nv=llama_vocab_n_tokens(vocab);state_logits.assign(logits,logits+nv);
            size_t bytes=llama_state_get_size(ctx);if(!bytes || bytes>size_t(4)*1024*1024*1024)return 1;
            std::vector<uint8_t> state(bytes);
            size_t saved=llama_state_get_data(ctx,state.data(),state.size());if(!saved || saved>state.size())return 1;
            state.resize(saved);
            llama_token next=std::max_element(state_logits.begin(),state_logits.end())-state_logits.begin();
            float *first=process(&next,1,tokens.size());if(!first)return 1;std::vector<float> expected(first,first+nv);
            if(llama_state_set_data(ctx,state.data(),state.size())!=state.size())return 1;
            float *second=process(&next,1,tokens.size());if(!second || std::memcmp(expected.data(),second,nv*sizeof(float)))return 1;
            if(llama_state_set_data(ctx,state.data(),state.size())!=state.size())return 1;
            logits=state_logits.data();
            std::fprintf(stderr,"K200STATE bytes=%zu next_token=%d restored_next_logits_bitwise=1\n",state.size(),next);
        }
        setenv("K200_PROFILE_PHASE",trial<0?"warmup.tg":"tg",1);
        FILE *dump=nullptr;
        if(argc>4 && trial>=0 && !std::getenv("K200_NO_DUMP")){
            std::string file=argv[4];if(lengths.size()>1)file+=".p"+std::to_string(length);if(trial)file+="."+std::to_string(trial);
            dump=std::fopen(file.c_str(),"wb");if(!dump)return 1;
        }
        std::vector<llama_token>generated;int nv=llama_vocab_n_tokens(vocab);
        for(int i=0;i<predict;++i){
            llama_token best=0;
            {
            K200_DIAG_SCOPE(scope,"harness","greedy_finite_dump_detokenize");
            for(int j=0;j<nv;++j){if(!std::isfinite(logits[j])){std::fprintf(stderr,"nonfinite logits\n");return 1;}if(logits[j]>logits[best])best=j;}
            if(dump && std::fwrite(logits,sizeof(float),nv,dump)!=(size_t)nv)return 1;
            generated.push_back(best);
            if(trial>=0){std::vector<char>piece(4096);int size=llama_token_to_piece(vocab,best,piece.data(),piece.size(),0,true);if(size<0)return 1;std::fwrite(piece.data(),1,size,stdout);}
            }
            if(teacher.empty() && llama_vocab_is_eog(vocab,best) && !std::getenv("K200_IGNORE_EOS"))break;
            if(i+1==predict)break;
            llama_token next=teacher.empty()?best:teacher[i];
            start=Clock::now();logits=process(&next,1,tokens.size()+i);tg_ms+=elapsed(start);++decode_steps;
            if(!logits)return 1;
        }
        if(dump)std::fclose(dump);
        if(trial>=0){
            if(task_set) {
                std::string answer;
                for(auto token:generated){std::vector<char>part(4096);int n=llama_token_to_piece(vocab,token,part.data(),part.size(),0,true);if(n<0)return 1;answer.append(part.data(),n);}
                std::printf("\nK200TASK {\"id\":\"%s\",\"text\":\"",task.id.c_str());
                for(unsigned char c:answer){if(c=='"' || c=='\\')std::printf("\\%c",c);else if(c<32)std::printf("\\u%04x",c);else std::putchar(c);}
                std::printf("\"}\n");std::fflush(stdout);
            }
            std::printf("\n");std::fprintf(stderr,"TOKENS mode=%s",mode);for(auto id:generated)std::fprintf(stderr," %d",id);std::fprintf(stderr,"\n");
            rusage cpu_end{};getrusage(RUSAGE_SELF,&cpu_end);
            double user_ms=(cpu_end.ru_utime.tv_sec-cpu_start.ru_utime.tv_sec)*1000.+(cpu_end.ru_utime.tv_usec-cpu_start.ru_utime.tv_usec)/1000.;
            double sys_ms=(cpu_end.ru_stime.tv_sec-cpu_start.ru_stime.tv_sec)*1000.+(cpu_end.ru_stime.tv_usec-cpu_start.ru_stime.tv_usec)/1000.;
            std::fprintf(stderr,"K200CPU trial=%d user_ms=%.6f sys_ms=%.6f percent=%.3f\n",trial,user_ms,sys_ms,100*(user_ms+sys_ms)/(pp_ms+tg_ms));
            std::fprintf(stderr,"K200PERF trial=%d pp_ms=%.6f pp_tokens=%zu tg_ms=%.6f tg_steps=%d first_process_ms=%.6f\n",trial,pp_ms,tokens.size(),tg_ms,decode_steps,first_ms);
            llama_perf_context_print(ctx);
        }
    }
    }
    if(auto emit=(void(*)())dlsym(RTLD_DEFAULT,"k200_profile_emit"))emit();
    if(auto memory=(void(*)())dlsym(RTLD_DEFAULT,"k200_profile_memory"))memory();
    for(const auto & item:trace.values)std::fprintf(stderr,"K200OP key=%s calls=%zu ms=%.6f\n",item.first.c_str(),item.second.calls,item.second.ms);
    rusage memory{};getrusage(RUSAGE_SELF,&memory);std::fprintf(stderr,"K200RSS peak_kib=%ld\n",memory.ru_maxrss);

    // Explicit lifecycle regression only; outside normal performance regions.
    if(std::getenv("K200_CHECK_KV_MUTATIONS")) {
        auto mem=llama_get_memory(ctx);int nv=llama_vocab_n_tokens(vocab);
        size_t bytes=llama_state_get_size(ctx);std::vector<uint8_t> state(bytes);
        size_t saved=llama_state_get_data(ctx,state.data(),state.size());if(!saved||saved>state.size())return 1;state.resize(saved);
        std::string file=std::string(argv[4])+".lifecycle";FILE *f=std::fopen(file.c_str(),"wb");if(!f)return 1;
        auto step=[&](const char *tag,int position,int token){llama_token value=token;float *row=process(&value,1,position);if(!row)return false;for(int j=0;j<nv;++j)if(!std::isfinite(row[j]))return false;if(std::fwrite(row,4,nv,f)!=(size_t)nv)return false;std::fprintf(stderr,"KV_LIFECYCLE case=%s position=%d pass=1\n",tag,position);return true;};
        int end=llama_memory_seq_pos_max(mem,0);
        if(!llama_memory_seq_rm(mem,0,end,-1)||!step("overwrite",end,42))return 1;
        int begin=end-7;if(begin<0)return 1;if(!llama_memory_seq_rm(mem,0,begin,-1))return 1;
        for(int j=0;j<8;++j)if(!step("rollback-reuse",begin+j,43+j))return 1;
        if(std::getenv("K200_CHECK_KV_SHIFT")) {
            if(!llama_memory_can_shift(mem)||!llama_memory_seq_rm(mem,0,0,32))return 1;
            llama_memory_seq_add(mem,0,32,-1,-32);if(!step("context-shift",end-31,91))return 1;
        }else std::fprintf(stderr,"KV_LIFECYCLE_SKIP case=context-shift reason=baseline-F16-inplace-RoPE-unsupported\n");
        if(llama_state_set_data(ctx,state.data(),state.size())!=state.size()||!step("restore",end+1,92))return 1;
        llama_memory_clear(mem,false);if(!step("clear-reuse",0,93))return 1;
        llama_memory_clear(mem,true);if(!step("zero-reuse",0,94))return 1;
        std::fclose(f);
    }

    if(task_set)std::fprintf(stderr,"K200CASE_END id=%s\n",task.id.c_str());
    } // prompt set; each trial above clears KV before PP
    setenv("K200_PROFILE_PHASE","cleanup",1);
    llama_batch_ext_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();
    return 0;
}

int main(int argc,char **argv) {
    int reloads=setting("K200_MODEL_RELOADS",1);
    for(int i=0;i<reloads;++i) {
        std::fprintf(stderr,"K200RELOAD index=%d\n",i);
        int rc=smoke_run(argc,argv);if(rc)return rc;
    }
    return 0;
}

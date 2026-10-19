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
int main(int argc,char **argv) {
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
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.no_perf=false;
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
    std::fprintf(stderr,"K200INIT backend_ms=%.3f model_ms=%.3f context_ms=%.3f ctx=%u batch=%u ubatch=%u prompt_tokens=%zu positions=%d teacher_forced=%d\n",backend_ms,model_ms,context_ms,cp.n_ctx,cp.n_batch,cp.n_ubatch,tokens.size(),predict,!teacher.empty());
    auto *batch=llama_batch_ext_init(ctx);
    auto process=[&](const llama_token *ids,int count,int pos){
        llama_batch_ext_clear(batch);
        for(int i=0;i<count;++i){int j=llama_batch_ext_add_token(batch,0,ids[i]);llama_pos p=pos+i;llama_batch_ext_set_pos(batch,j,&p);}
        llama_batch_ext_set_output_logits(batch,count-1,true);
        if(llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,batch))return (float *)nullptr;
        return llama_get_logits_ith(ctx,-1);
    };
    int repeats=setting("K200_REPEATS",1),warmup=std::getenv("K200_WARMUP")?1:0;
    for(int trial=-warmup;trial<repeats;++trial){
        setenv("K200_PROFILE_PHASE","clear",1);
        if(trial!=-warmup)llama_memory_clear(llama_get_memory(ctx),true);
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
        setenv("K200_PROFILE_PHASE",trial<0?"warmup.tg":"tg",1);
        FILE *dump=nullptr;
        if(argc>4 && trial>=0){
            std::string file=argv[4];if(trial)file+="."+std::to_string(trial);
            dump=std::fopen(file.c_str(),"wb");if(!dump)return 1;
        }
        std::vector<llama_token>generated;int nv=llama_vocab_n_tokens(vocab);
        for(int i=0;i<predict;++i){
            llama_token best=0;
            for(int j=0;j<nv;++j){if(!std::isfinite(logits[j])){std::fprintf(stderr,"nonfinite logits\n");return 1;}if(logits[j]>logits[best])best=j;}
            if(dump && std::fwrite(logits,sizeof(float),nv,dump)!=(size_t)nv)return 1;
            generated.push_back(best);
            if(trial>=0){std::vector<char>piece(4096);int size=llama_token_to_piece(vocab,best,piece.data(),piece.size(),0,true);if(size<0)return 1;std::fwrite(piece.data(),1,size,stdout);}
            if(teacher.empty() && llama_vocab_is_eog(vocab,best) && !std::getenv("K200_IGNORE_EOS"))break;
            if(i+1==predict)break;
            llama_token next=teacher.empty()?best:teacher[i];
            start=Clock::now();logits=process(&next,1,tokens.size()+i);tg_ms+=elapsed(start);++decode_steps;
            if(!logits)return 1;
        }
        if(dump)std::fclose(dump);
        if(trial>=0){
            std::printf("\n");std::fprintf(stderr,"TOKENS mode=%s",mode);for(auto id:generated)std::fprintf(stderr," %d",id);std::fprintf(stderr,"\n");
            rusage cpu_end{};getrusage(RUSAGE_SELF,&cpu_end);
            double user_ms=(cpu_end.ru_utime.tv_sec-cpu_start.ru_utime.tv_sec)*1000.+(cpu_end.ru_utime.tv_usec-cpu_start.ru_utime.tv_usec)/1000.;
            double sys_ms=(cpu_end.ru_stime.tv_sec-cpu_start.ru_stime.tv_sec)*1000.+(cpu_end.ru_stime.tv_usec-cpu_start.ru_stime.tv_usec)/1000.;
            std::fprintf(stderr,"K200CPU trial=%d user_ms=%.6f sys_ms=%.6f percent=%.3f\n",trial,user_ms,sys_ms,100*(user_ms+sys_ms)/(pp_ms+tg_ms));
            std::fprintf(stderr,"K200PERF trial=%d pp_ms=%.6f pp_tokens=%zu tg_ms=%.6f tg_steps=%d first_process_ms=%.6f\n",trial,pp_ms,tokens.size(),tg_ms,decode_steps,first_ms);
            llama_perf_context_print(ctx);
        }
    }
    if(auto emit=(void(*)())dlsym(RTLD_DEFAULT,"k200_profile_emit"))emit();
    if(auto memory=(void(*)())dlsym(RTLD_DEFAULT,"k200_profile_memory"))memory();
    llama_batch_ext_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();
}

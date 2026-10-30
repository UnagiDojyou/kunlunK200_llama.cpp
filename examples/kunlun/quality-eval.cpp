#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

// Standalone evaluation only. Each independent context scores its next tokens;
// every configuration consumes the same IDs, positions and context boundaries.
static bool sample_position(int p,int stride) {
    return stride==1 || p%stride==0 || p==127 || p==128 || p==255 || p==256 || p==257 || p==511;
}
static int setting(const char *name,int fallback,int lo,int hi) {
    const char *s=std::getenv(name);if(!s)return fallback;
    char *end=nullptr;long n=std::strtol(s,&end,10);
    if(end==s || *end || n<lo || n>hi){std::fprintf(stderr,"invalid %s\n",name);std::exit(2);}return n;
}
static std::vector<int> top5(const float *x,int nv) {
    std::vector<int> ids(nv);for(int i=0;i<nv;++i)ids[i]=i;
    std::partial_sort(ids.begin(),ids.begin()+5,ids.end(),[&](int a,int b){return x[a]>x[b] || (x[a]==x[b] && a<b);});ids.resize(5);return ids;
}
struct Reference {
    FILE *file=nullptr;
    std::vector<float> row;
    double squared=0,energy=0,kl=0,max_kl=0,max_abs=0,true_lp=0,max_true_lp=0;
    size_t equal=0,positions=0,top5_overlap=0,top5_equal=0,near_tie_changes=0,large_kl_rows=0;
    int worst_doc=-1,worst_pos=-1,worst_target=-1;
    double worst_gap=0,worst_lp=0;
    std::vector<int> worst_ref,worst_actual;
    void open(const char *path,int nv,int count,int doc_size,int stride) {
        if(!std::strcmp(path,"-"))return;
        file=std::fopen(path,"rb");if(!file)std::exit(2);
        if(stride==1){int h[2];if(std::fread(h,4,2,file)!=2 || h[0]!=nv || h[1]!=count)std::exit(2);}
        else {int h[6];if(std::fread(h,4,6,file)!=6 || h[0]!=0x4b515331 || h[1]!=1 || h[2]!=nv || h[3]!=count || h[4]!=doc_size || h[5]!=stride)std::exit(2);}
        row.resize(nv);
    }
    void add(const float *actual,int nv,int target,int doc,int pos) {
        if(!file)return;
        if(std::fread(row.data(),4,nv,file)!=(size_t)nv)std::exit(2);
        double ma=*std::max_element(row.begin(),row.end()),mb=*std::max_element(actual,actual+nv),sa=0,sb=0;
        for(int j=0;j<nv;++j){sa+=std::exp(row[j]-ma);sb+=std::exp(actual[j]-mb);double d=double(actual[j])-row[j];squared+=d*d;energy+=double(row[j])*row[j];max_abs=std::max(max_abs,std::abs(d));}
        double la=ma+std::log(sa),lb=mb+std::log(sb),k=0;
        for(int j=0;j<nv;++j)k+=std::exp(row[j]-la)*((row[j]-la)-(actual[j]-lb));
        auto a=top5(row.data(),nv),b=top5(actual,nv);bool same=a[0]==b[0];size_t overlap=0;
        for(int id:a)overlap+=std::find(b.begin(),b.end(),id)!=b.end();
        double delta=(actual[target]-lb)-(row[target]-la),gap=row[a[0]]-row[a[1]];
        true_lp+=delta;max_true_lp=std::max(max_true_lp,std::abs(delta));
        if(k>max_kl || !positions){max_kl=k;worst_doc=doc;worst_pos=pos;worst_target=target;worst_gap=gap;worst_lp=delta;worst_ref=a;worst_actual=b;}
        kl+=k;++positions;equal+=same;top5_overlap+=overlap;top5_equal+=overlap==5;
        near_tie_changes+=!same && gap<=.1;large_kl_rows+=k>.1;
    }
    void print(int id) {
        if(!file)return;
        if(std::fgetc(file)!=EOF || !positions)std::exit(2);
        std::fclose(file);
        std::printf("QUALITY reference=%d positions=%zu nmse=%.12g mean_KL=%.12g max_KL=%.12g max_abs=%.12g greedy_equal=%zu top5_mean_overlap=%.12g top5_set_equal=%zu near_tie_changes_gap_le_0_1=%zu large_KL_gt_0_1=%zu mean_true_logp_delta=%.12g max_abs_true_logp_delta=%.12g\n",id,positions,squared/std::max(energy,1e-30),kl/positions,max_kl,max_abs,equal,double(top5_overlap)/positions,top5_equal,near_tie_changes,large_kl_rows,true_lp/positions,max_true_lp);
        std::printf("QUALITY_WORST reference=%d doc=%d position=%d target=%d KL=%.12g ref_top2_gap=%.12g true_logp_delta=%.12g ref_top5=",id,worst_doc,worst_pos,worst_target,max_kl,worst_gap,worst_lp);
        for(int i=0;i<5;++i)std::printf("%s%d",i?",":"",worst_ref[i]);
        std::printf(" actual_top5=");
        for(int i=0;i<5;++i)std::printf("%s%d",i?",":"",worst_actual[i]);
        std::printf("\n");
    }
};

int main(int argc,char **argv) {
    if(argc<6){std::fprintf(stderr,"usage: model tokenize text count ids | model cpu|none|layer|row ids ctx dump|- [reference|-] [reference2|-]\n");return 2;}
    bool tokenize=!std::strcmp(argv[2],"tokenize"),cpu=tokenize || !std::strcmp(argv[2],"cpu");
    if(!cpu && std::strcmp(argv[2],"none") && std::strcmp(argv[2],"layer") && std::strcmp(argv[2],"row"))return 2;
    llama_backend_init();ggml_backend_dev_t devices[3]={nullptr,nullptr,nullptr};
    if(!cpu){devices[0]=ggml_backend_dev_by_name("KUNLUN0");if(std::strcmp(argv[2],"none"))devices[1]=ggml_backend_dev_by_name("KUNLUN1");if(!devices[0])return 1;}
    float split[2]={1,1};auto mp=llama_model_default_params();mp.vocab_only=tokenize;mp.devices=devices;mp.n_gpu_layers=cpu?0:99;mp.tensor_split=split;
    mp.split_mode=!std::strcmp(argv[2],"row")?LLAMA_SPLIT_MODE_ROW:!std::strcmp(argv[2],"layer")?LLAMA_SPLIT_MODE_LAYER:LLAMA_SPLIT_MODE_NONE;
    auto *model=llama_model_load_from_file(argv[1],mp);if(!model)return 1;auto *vocab=llama_model_get_vocab(model);int nv=llama_vocab_n_tokens(vocab);
    std::vector<llama_token> ids;
    if(tokenize){
        std::ifstream in(argv[3],std::ios::binary);if(!in)return 2;std::string text((std::istreambuf_iterator<char>(in)),{});
        int count=-llama_tokenize(vocab,text.data(),text.size(),nullptr,0,true,true);if(count<1)return 2;ids.resize(count);
        if(llama_tokenize(vocab,text.data(),text.size(),ids.data(),count,true,true)!=count)return 2;
        int cap=std::atoi(argv[4]);if(cap<1 || count<cap)return 2;ids.resize(cap);std::ofstream out(argv[5]);if(!out)return 2;
        for(auto id:ids)out<<id<<'\n';
        unsigned long long fingerprint=1469598103934665603ull;
        for(int t=0;t<nv;++t){const unsigned char *q=(const unsigned char*)llama_vocab_get_text(vocab,t);while(*q){fingerprint^=*q++;fingerprint*=1099511628211ull;}fingerprint^=255;fingerprint*=1099511628211ull;}
        std::printf("TOKENIZED tokens=%zu available=%d vocab=%d bos=%d eos=%d vocabulary_fingerprint=%llu\n",ids.size(),count,nv,llama_vocab_bos(vocab),llama_vocab_eos(vocab),fingerprint);
        llama_model_free(model);llama_backend_free();return 0;
    }
    {std::ifstream in(argv[3]);long id;while(in>>id){if(id<0 || id>=nv)return 2;ids.push_back(id);}if(!in.eof())return 2;}
    int ctx_size=std::atoi(argv[4]);if(ctx_size<32 || ctx_size>1024 || ids.size()<(size_t)ctx_size || ids.size()%ctx_size)return 2;
    int scored=ids.size()/ctx_size*(ctx_size-1);
    auto cp=llama_context_default_params();int batch_size=8;
    if(const char *setting=std::getenv("K200_QUALITY_BATCH")) {
        char *end=nullptr;long value=std::strtol(setting,&end,10);
        if(end==setting || *end || value<1 || value>ctx_size)return 2;
        batch_size=value;
    }
    int stride=setting("K200_QUALITY_SAMPLE_STRIDE",1,1,1024);
    cp.n_ctx=setting("K200_QUALITY_CONTEXT",ctx_size,ctx_size,32768);cp.n_batch=batch_size;cp.n_ubatch=setting("K200_QUALITY_UBATCH",batch_size,1,batch_size);cp.n_threads=6;cp.n_threads_batch=6;
    cp.offload_kqv=std::getenv("GGML_KUNLUN_CPU_KV")==nullptr;cp.flash_attn_type=(std::getenv("K200_FLASH")&&!std::strcmp(std::getenv("K200_FLASH"),"1"))?LLAMA_FLASH_ATTN_TYPE_ENABLED:LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto *ctx=llama_init_from_model(model,cp);if(!ctx)return 1;auto *batch=llama_batch_ext_init(ctx);
    Reference references[2];if(argc>6)references[0].open(argv[6],nv,scored,ctx_size,stride);if(argc>7)references[1].open(argv[7],nv,scored,ctx_size,stride);
    FILE *dump=nullptr;if(std::strcmp(argv[5],"-")){dump=std::fopen(argv[5],"wb");if(!dump)return 2;if(stride==1){int h[2]={nv,scored};if(std::fwrite(h,4,2,dump)!=2)return 2;}else{int h[6]={0x4b515331,1,nv,scored,ctx_size,stride};if(std::fwrite(h,4,6,dump)!=6)return 2;}}
    double nll=0;size_t count=0;
    std::vector<std::pair<std::string,std::string>> docs;
    if(const char *path=std::getenv("K200_QUALITY_DOCS")){std::ifstream f(path);std::string id,category;while(f>>id>>category)docs.emplace_back(id,category);if(!f.eof() || docs.size()!=ids.size()/ctx_size)return 2;}
    for(size_t offset=0;offset<ids.size();offset+=ctx_size){
        double before_nll=nll;size_t before_count=count;
        if(offset)llama_memory_clear(llama_get_memory(ctx),true);
        for(int position=0;position<ctx_size-1;){
            int nb=std::min(batch_size,ctx_size-1-position);llama_batch_ext_clear(batch);
            for(int j=0;j<nb;++j){int i=llama_batch_ext_add_token(batch,0,ids[offset+position+j]);llama_pos p=position+j;llama_batch_ext_set_pos(batch,i,&p);llama_batch_ext_set_output_logits(batch,i,true);}
            if(llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,batch))return 1;
            for(int j=0;j<nb;++j){
                const float *logits=llama_get_logits_ith(ctx,j);if(!logits)return 1;
                double maximum=-INFINITY;for(int v=0;v<nv;++v){if(!std::isfinite(logits[v]))return 1;maximum=std::max(maximum,double(logits[v]));}
                double sum=0;for(int v=0;v<nv;++v)sum+=std::exp(logits[v]-maximum);
                nll+=maximum+std::log(sum)-logits[ids[offset+position+j+1]];++count;
                if(sample_position(position+j,stride)) {
                    if(dump && std::fwrite(logits,4,nv,dump)!=(size_t)nv)return 2;
                    for(auto &r:references)r.add(logits,nv,ids[offset+position+j+1],offset/ctx_size,position+j);
                }
            }
            position+=nb;
        }
        double dnll=(nll-before_nll)/(count-before_count);auto doc=offset/ctx_size;
        std::printf("DOCUMENT index=%zu id=%s category=%s input_tokens=%d scored=%zu mean_nll=%.12g perplexity=%.12g\n",doc,docs.empty()?"unspecified":docs[doc].first.c_str(),docs.empty()?"unspecified":docs[doc].second.c_str(),ctx_size,count-before_count,dnll,std::exp(dnll));
        std::fprintf(stderr,"EVAL progress=%zu/%zu nll=%.9f\n",count,(size_t)scored,nll/count);
    }
    if(dump)std::fclose(dump);
    for(int i=0;i<2;++i)references[i].print(i);
    std::printf("CORPUS input_tokens=%zu scored=%zu context=%d batch=%d mean_nll=%.12g perplexity=%.12g\n",ids.size(),count,ctx_size,batch_size,nll/count,std::exp(nll/count));
    llama_batch_ext_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();
}

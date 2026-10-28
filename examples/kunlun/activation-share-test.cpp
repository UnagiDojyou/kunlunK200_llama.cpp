#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

static std::vector<float> run(int pd,int m,int kind,bool share,bool row) {
    setenv("GGML_KUNLUN_Q4_ACT_SHARE",share?"1":"0",1);
    auto * backend=ggml_backend_init_by_name(pd?"KUNLUN1":"KUNLUN0",nullptr);GGML_ASSERT(backend);
    auto * ctx=ggml_init({2*1024*1024,nullptr,true});GGML_ASSERT(ctx);
    auto * wc=ggml_init({1024*1024,nullptr,true});GGML_ASSERT(wc);
    const int k=512,n=512;
    auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,m);ggml_set_name(x,"shared_input");
    ggml_tensor * weights[3],* ys[3];
    for(int i=0;i<3;++i)weights[i]=ggml_new_tensor_2d(wc,GGML_TYPE_Q4_0,k,n);
    auto * graph=ggml_new_graph_custom(ctx,128,false);
    ys[0]=ggml_mul_mat(ctx,weights[0],x);ggml_build_forward_expand(graph,ys[0]);
    if(kind==1) {
        // A real write between consumers aliases x but has a different tensor object.
        auto * changed=ggml_scale_inplace(ctx,x,2.0f);ggml_set_name(changed,"alias_write");
        ggml_build_forward_expand(graph,changed);
    }
    ggml_tensor * mixed=nullptr,* mx=nullptr,* mw=nullptr;
    if(kind==3) {
        mx=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,1024,m);
        mw=ggml_new_tensor_2d(wc,GGML_TYPE_F32,1024,n);
        mixed=ggml_mul_mat(ctx,mw,mx);ggml_set_name(mixed,"mixed_F32_staging_resize");
        ggml_build_forward_expand(graph,mixed);
    }
    auto * second=kind==2?ggml_view_2d(ctx,x,k,m,x->nb[1],0):x;
    for(int i=1;i<3;++i){ys[i]=ggml_mul_mat(ctx,weights[i],second);ggml_build_forward_expand(graph,ys[i]);}
    auto * buf=ggml_backend_alloc_ctx_tensors(ctx,backend);GGML_ASSERT(buf);
    ggml_backend_buffer_t wb=nullptr;
    if(row) {
        auto reg=ggml_backend_reg_by_name("KUNLUN");
        auto split=(ggml_backend_split_buffer_type_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_split_buffer_type");GGML_ASSERT(split);
        float fractions[2]={1,1};wb=ggml_backend_alloc_ctx_tensors_from_buft(wc,split(pd,fractions));
    } else wb=ggml_backend_alloc_ctx_tensors(wc,backend);
    GGML_ASSERT(wb);
    if(mw){std::vector<float> data((size_t)1024*n,.03125f);ggml_backend_tensor_set(mw,data.data(),0,data.size()*4);}
    for(int j=0;j<3;++j) {
        std::vector<unsigned char> w(ggml_nbytes(weights[j]));ggml_fp16_t scale=ggml_fp32_to_fp16(.03125f);
        for(size_t b=0;b<w.size()/18;++b){std::memcpy(w.data()+b*18,&scale,2);for(int a=0;a<16;++a)w[b*18+2+a]=(unsigned char)((b+a*17+j*31)%256);}
        ggml_backend_tensor_set(weights[j],w.data(),0,w.size());
    }
    std::vector<float> all;
    for(int generation=0;generation<3;++generation) {
        std::vector<float> input((size_t)k*m);
        for(size_t i=0;i<input.size();++i)input[i]=(int((i*13+generation*73)%503)-251)*.00390625f;
        // Same graph and same input address, new contents on every invocation.
        ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
        if(mx){std::vector<float> other((size_t)1024*m,(generation+1)*.0625f);ggml_backend_tensor_set(mx,other.data(),0,other.size()*4);}
        GGML_ASSERT(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
        if(mixed){size_t at=all.size();all.resize(at+(size_t)n*m);ggml_backend_tensor_get(mixed,all.data()+at,0,(size_t)n*m*4);}
        for(auto * y:ys){size_t at=all.size();all.resize(at+(size_t)n*m);ggml_backend_tensor_get(y,all.data()+at,0,(size_t)n*m*4);}
    }
    ggml_backend_buffer_free(wb);ggml_backend_buffer_free(buf);ggml_free(wc);ggml_free(ctx);ggml_backend_free(backend);return all;
}
int main(int argc,char **argv) {
    GGML_ASSERT(argc==2);int pd=std::atoi(argv[1]);GGML_ASSERT(pd>=0&&pd<2);
    setenv("GGML_KUNLUN_COMMON_Q4","1",1);ggml_backend_load_all();
    for(bool row:{false,true})for(int m:{1,9,129})for(int kind:{0,1,2,3}) {
        auto old=run(pd,m,kind,false,row),shared=run(pd,m,kind,true,row);
        GGML_ASSERT(old.size()==shared.size() && !std::memcmp(old.data(),shared.data(),old.size()*4));
        std::printf("ACTIVATION_GRAPH pd=%d row=%d M=%d case=%d generations=3 bitwise=1\n",pd,row,m,kind);
    }
}

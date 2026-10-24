#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static std::vector<float> evaluate(ggml_backend_dev_t dev,ggml_type type,int w,int rows,bool packed,bool swapped,int gap,bool nd=false) {
    auto backend=ggml_backend_dev_init(dev,nullptr);GGML_ASSERT(backend);
    auto *ctx=ggml_init({2*1024*1024,nullptr,true});GGML_ASSERT(ctx);
    int aw=w*(packed?2:1),bs=ggml_type_size(type);
    auto *a_store=ggml_new_tensor_2d(ctx,type,aw+gap,rows);
    auto *a=ggml_view_2d(ctx,a_store,aw,rows,a_store->nb[1],0);
    auto *b_store=packed?nullptr:ggml_new_tensor_2d(ctx,type,w+gap,rows);
    auto *b=packed?nullptr:ggml_view_2d(ctx,b_store,w,rows,b_store->nb[1],0);
    if(nd){a=ggml_reshape_3d(ctx,a,aw,3,2);if(b)b=ggml_reshape_3d(ctx,b,w,3,2);}
    auto *y=packed?(swapped?ggml_swiglu_swapped(ctx,a):ggml_swiglu(ctx,a)):ggml_swiglu_split(ctx,a,b);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev,y));
    auto *graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);GGML_ASSERT(buffer);
    std::vector<unsigned char>av(ggml_nbytes(a_store)),bv(b?ggml_nbytes(b_store):0);
    auto put=[&](unsigned char *p,float v){if(bs==4)std::memcpy(p,&v,4);else{auto h=ggml_fp32_to_fp16(v);std::memcpy(p,&h,2);}};
    for(int r=0;r<rows;++r)for(int j=0;j<w;++j) {
        float g=((j*31+r*7)%1025-512)*.015625f,u=((j*17+r*13)%513-256)*.03125f;
        put(av.data()+r*a_store->nb[1]+(j+(packed&&swapped?w:0))*bs,g);
        put(packed?av.data()+r*a_store->nb[1]+(j+(swapped?0:w))*bs:bv.data()+r*b_store->nb[1]+j*bs,u);
    }
    ggml_backend_tensor_set(a_store,av.data(),0,av.size());if(b)ggml_backend_tensor_set(b_store,bv.data(),0,bv.size());
    GGML_ASSERT(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
    std::vector<unsigned char>raw(ggml_nbytes(y));ggml_backend_tensor_get(y,raw.data(),0,raw.size());
    std::vector<float>out((size_t)w*rows);
    for(size_t i=0;i<out.size();++i){if(bs==4)std::memcpy(&out[i],raw.data()+i*4,4);else{ggml_fp16_t h;std::memcpy(&h,raw.data()+i*2,2);out[i]=ggml_fp16_to_fp32(h);}}
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);return out;
}
int main(int argc,char **argv) {
    int pd=argc>1?std::atoi(argv[1]):0;GGML_ASSERT(pd==0 || pd==1);
    ggml_backend_load_all();char name[32];std::snprintf(name,sizeof(name),"KUNLUN%d",pd);
    auto *gpu=ggml_backend_dev_by_name(name),*cpu=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);GGML_ASSERT(gpu && cpu);
    for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16})for(int mode=0;mode<5;++mode) {
        int w=mode?257:11008,rows=mode==4?6:mode?7:32,gap=mode==3?16:0;
        bool packed=mode==1 || mode==2,swapped=mode==2;
        auto ref=evaluate(cpu,type,w,rows,packed,swapped,gap,mode==4),out=evaluate(gpu,type,w,rows,packed,swapped,gap,mode==4);
        double sq=0,en=0,maxerr=0;
        for(size_t i=0;i<out.size();++i){GGML_ASSERT(std::isfinite(out[i]));double d=out[i]-ref[i];sq+=d*d;en+=double(ref[i])*ref[i];maxerr=std::max(maxerr,std::abs(d));}
        double nmse=sq/std::max(en,1e-30);GGML_ASSERT(nmse<(type==GGML_TYPE_F32?1e-12:1e-6));
        std::printf("GGML_SWIGLU pd=%d type=%s mode=%d width=%d rows=%d nmse=%.12g max_abs=%.12g PASS\n",pd,ggml_type_name(type),mode,w,rows,nmse,maxerr);
    }
    auto *ctx=ggml_init({128*1024,nullptr,true});auto *a=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,256,8),*b=ggml_dup_tensor(ctx,a);
    auto *y=ggml_swiglu_split(ctx,a,b);GGML_ASSERT(ggml_backend_dev_supports_op(gpu,y));
    auto saved=a->nb[0];a->nb[0]=8;GGML_ASSERT(!ggml_backend_dev_supports_op(gpu,y));a->nb[0]=saved;
    GGML_ASSERT(!ggml_backend_dev_supports_op(gpu,ggml_geglu_split(ctx,a,b)));
    auto *mis=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,257,3);
    auto *mv=ggml_view_2d(ctx,mis,256,2,mis->nb[1],2);
    GGML_ASSERT(!ggml_backend_dev_supports_op(gpu,ggml_swiglu_split(ctx,mv,mv)));
    auto *bf=ggml_new_tensor_2d(ctx,GGML_TYPE_BF16,256,8);GGML_ASSERT(!ggml_backend_dev_supports_op(gpu,ggml_swiglu_split(ctx,bf,bf)));
    auto *nd=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,256,2,2);auto *ny=ggml_swiglu_split(ctx,nd,nd);
    GGML_ASSERT(ggml_backend_dev_supports_op(gpu,ny));nd->nb[2]+=4;GGML_ASSERT(!ggml_backend_dev_supports_op(gpu,ny));
    setenv("GGML_KUNLUN_SWIGLU","pp",1);GGML_ASSERT(ggml_backend_dev_supports_op(gpu,y));
    auto *single=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,256,1);
    GGML_ASSERT(!ggml_backend_dev_supports_op(gpu,ggml_swiglu_split(ctx,single,single)));
    setenv("GGML_KUNLUN_SWIGLU","0",1);GGML_ASSERT(!ggml_backend_dev_supports_op(gpu,y));setenv("GGML_KUNLUN_SWIGLU","1",1);
    std::printf("GGML_SWIGLU unsupported stride/BF16/GEGLU/disabled CPU placement PASS\n");ggml_free(ctx);
}

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    int k = argc>1 ? atoi(argv[1]) : 4096;
    int n = argc>2 ? atoi(argv[2]) : 4096;
    int batch = argc>3 ? atoi(argv[3]) : 1;
    bool split = argc>4 && atoi(argv[4]);
    ggml_type type = argc>5 ? (ggml_type)atoi(argv[5]) : GGML_TYPE_Q4_0;
    ggml_backend_load_all();
    int device = argc>6 ? atoi(argv[6]) : 0;
    char device_name[32]; snprintf(device_name,sizeof(device_name),"KUNLUN%d",device);
    auto * dev = ggml_backend_dev_by_name(device_name);
    GGML_ASSERT(dev && k>0 && n>0 && batch>0 && k%32==0);
    auto backend = ggml_backend_dev_init(dev,nullptr);
    auto * ctx = ggml_init({8*1024*1024,nullptr,true});
    auto * w = ggml_new_tensor_2d(ctx,type,k,n);
    auto * x = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,batch);
    auto * y = ggml_mul_mat(ctx,w,x);
    auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph,y);
    auto buft = ggml_backend_dev_buffer_type(dev);
    float fractions[2]={1,1};
    if (const char * setting=std::getenv("GGML_KUNLUN_TEST_SPLIT_FRACTION")) {
        char * end=nullptr;
        double fraction=std::strtod(setting,&end);
        GGML_ASSERT(end != setting && *end == '\0' && std::isfinite(fraction) && fraction >= 0 && fraction <= 1);
        fractions[0]=fraction; fractions[1]=1-fraction;
    }
    if(split) {
        auto fn=(ggml_backend_split_buffer_type_t)ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),"ggml_backend_split_buffer_type");
        buft=fn(device,fractions);
    }
    auto wb = ggml_backend_buft_alloc_buffer(buft,ggml_backend_buft_get_alloc_size(buft,w)+256);
    GGML_ASSERT(wb && ggml_backend_tensor_alloc(wb,w,ggml_backend_buffer_get_base(wb))==GGML_STATUS_SUCCESS);
    auto xb = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev),ggml_nbytes(x)+256);
    auto yb = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev),ggml_nbytes(y)+256);
    GGML_ASSERT(xb && yb);
    ggml_backend_tensor_alloc(xb,x,ggml_backend_buffer_get_base(xb));
    ggml_backend_tensor_alloc(yb,y,ggml_backend_buffer_get_base(yb));
    std::vector<float> wf((size_t)k*n), xf((size_t)k*batch), out((size_t)n*batch), ref(out.size());
    for(size_t i=0;i<wf.size();++i) wf[i]=.1f*sinf((float)(i%173)*.31f);
    for(size_t i=0;i<xf.size();++i) xf[i]=cosf((float)(i%137)*.23f);
    std::vector<char> packed(ggml_nbytes(w));
    ggml_quantize_chunk(type,wf.data(),packed.data(),0,n,k,nullptr);
    if(type==GGML_TYPE_F32) std::memcpy(wf.data(),packed.data(),packed.size());
    else ggml_get_type_traits(type)->to_float(packed.data(),wf.data(),wf.size());
    if (std::getenv("GGML_KUNLUN_TEST_CLEAR")) {
        ggml_backend_tensor_set(w,packed.data(),0,packed.size());
        ggml_backend_buffer_clear(wb,0);
        std::vector<char> cleared(packed.size(),1);
        ggml_backend_tensor_get(w,cleared.data(),0,cleared.size());
        for (char value : cleared) GGML_ASSERT(value == 0);
    }
    if (std::getenv("GGML_KUNLUN_TEST_CHUNKS")) {
        size_t first=packed.size()/3;
        ggml_backend_tensor_set(w,packed.data(),0,first);
        ggml_backend_tensor_set(w,packed.data()+first,first,packed.size()-first);
    } else ggml_backend_tensor_set(w,packed.data(),0,packed.size());
    std::vector<char> restored(packed.size());
    ggml_backend_tensor_get(w,restored.data(),0,restored.size());
    GGML_ASSERT(restored == packed);
    if (n > 1) {
        size_t bytes=ggml_row_size(type,k);
        auto * view=ggml_view_2d(ctx,w,k,1,bytes,bytes);
        GGML_ASSERT(ggml_backend_view_init(view)==GGML_STATUS_SUCCESS);
        std::vector<char> viewed(bytes);
        ggml_backend_tensor_get(view,viewed.data(),0,bytes);
        GGML_ASSERT(std::memcmp(viewed.data(),packed.data()+bytes,bytes)==0);
    }
    ggml_backend_tensor_set(x,xf.data(),0,xf.size()*4);
    auto represented=xf;
    const char *common_setting=std::getenv("GGML_KUNLUN_COMMON_Q4");
    bool common=common_setting && !std::strcmp(common_setting,"1") && type==GGML_TYPE_Q4_0 && k<=16384 && n<=32000;
    if(common && batch>1) for(int b=0;b<batch;++b) {
        float mx=0;for(int j=0;j<k;++j)mx=std::max(mx,std::abs(xf[(size_t)b*k+j]));if(mx==0)mx=1;
        float inv=127.f/mx,scale=mx/127.f;
        for(int j=0;j<k;++j){float v=xf[(size_t)b*k+j]*inv;int q=std::max(-127,std::min(127,(int)(v+(v>=0?.5f:-.5f))));represented[(size_t)b*k+j]=q*scale;}
    }
    for(int b=0;b<batch;++b) for(int r=0;r<n;++r) {
        double s=0; for(int j=0;j<k;++j)s+=(double)wf[(size_t)r*k+j]*represented[(size_t)b*k+j];
        ref[(size_t)b*n+r]=s;
    }
    for(int i=0;i<2;++i) GGML_ASSERT(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(y,out.data(),0,out.size()*4);
    const char * setting = std::getenv("GGML_KUNLUN_REQUANT_I8");
    bool requant = setting && std::strcmp(setting,"0") != 0 && type == GGML_TYPE_Q4_0;
    const char * native_setting=std::getenv("GGML_KUNLUN_NATIVE_Q4");
    bool native=native_setting && std::strcmp(native_setting,"1")==0 && type==GGML_TYPE_Q4_0 && k<=16384 && n<=90000000/k;
    native=native && !common;requant=requant && !native && !common;
    double maxerr=0,sqerr=0,sqref=0;
    for(size_t i=0;i<out.size();++i) {
        double err=std::abs(out[i]-ref[i]); maxerr=std::max(maxerr,err); sqerr+=err*err; sqref+=(double)ref[i]*ref[i];
        if(!std::isfinite(out[i]) || err>(requant ? .02 : .001)*(1+std::abs(ref[i]))) { fprintf(stderr,"FAIL idx=%zu got=%g ref=%g\n",i,out[i],ref[i]);return 1; }
    }
    if (common && sqerr/(sqref+1e-30) > (batch>1?1e-10:1e-6)) return 1;
    if (common) printf("Common Q4: Row A8 reference for M>1; two-component tolerance for M=1\n");
    if (requant && sqerr/(sqref+1e-30) > 1e-3) { fprintf(stderr,"FAIL requantization NMSE=%g\n",sqerr/(sqref+1e-30)); return 1; }
    if (native && sqerr/(sqref+1e-30) > 1e-6) return 1;
    if (native) printf("Native Q4 profile: original weights, CPU reference NMSE limit 1e-6\n");
    if (requant) printf("CDNN rowwise INT8 requantization: max per-output error 2%%*(1+abs(reference)), NMSE limit 1e-3\n");
    auto begin=std::chrono::steady_clock::now();
    const int runs=5;
    for(int i=0;i<runs;++i) GGML_ASSERT(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/runs;
    printf("PASS type=%s k=%d n=%d batch=%d split=%d elements=%zu max_abs=%g nmse=%g ms=%.4f weight_GBs=%.4f\n",ggml_type_name(type),k,n,batch,split,out.size(),maxerr,sqerr/(sqref+1e-30),ms,packed.size()/ms/1e6);
    ggml_backend_buffer_free(wb);ggml_backend_buffer_free(xb);ggml_backend_buffer_free(yb);ggml_free(ctx);ggml_backend_free(backend);
}

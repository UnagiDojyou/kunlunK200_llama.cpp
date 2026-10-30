// Diagnostic only: replay identical real activations through CPU/XPU GLU,
// then compare the unchanged library activation quantizer. Never used for timing LLMs.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <kunlun/k200.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>
static std::vector<float> read(const char *p) {
    std::ifstream f(p,std::ios::binary|std::ios::ate);GGML_ASSERT(f);
    auto n=f.tellg();GGML_ASSERT(n>0 && n%4==0);std::vector<float> v((size_t)n/4);f.seekg(0);f.read((char*)v.data(),n);GGML_ASSERT(f);return v;
}
static std::vector<float> run(ggml_backend_dev_t dev,const std::vector<float>&g,const std::vector<float>&u,int w,int bytes,double &ms) {
    auto backend=ggml_backend_dev_init(dev,nullptr);GGML_ASSERT(backend);
    if (ggml_backend_dev_type(dev)==GGML_BACKEND_DEVICE_TYPE_CPU) ggml_backend_cpu_set_n_threads(backend,6);
    auto *ctx=ggml_init({1024*1024,nullptr,true});GGML_ASSERT(ctx);
    auto type=bytes==4?GGML_TYPE_F32:GGML_TYPE_F16;
    auto *a=ggml_new_tensor_2d(ctx,type,w,g.size()/w),*b=ggml_dup_tensor(ctx,a),*y=ggml_swiglu_split(ctx,a,b);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev,y));auto *graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);GGML_ASSERT(buffer);
    std::vector<ggml_fp16_t> gh(g.size()),uh(u.size());
    if(bytes==2)for(size_t i=0;i<g.size();++i){gh[i]=ggml_fp32_to_fp16(g[i]);uh[i]=ggml_fp32_to_fp16(u[i]);}
    ggml_backend_tensor_set(a,bytes==4?(const void*)g.data():gh.data(),0,g.size()*bytes);
    ggml_backend_tensor_set(b,bytes==4?(const void*)u.data():uh.data(),0,u.size()*bytes);
    GGML_ASSERT(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
    std::vector<double> times;
    for(int i=0;i<3;++i){auto t=std::chrono::steady_clock::now();GGML_ASSERT(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);ggml_backend_synchronize(backend);times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());}
    std::sort(times.begin(),times.end());ms=times[1];
    std::vector<unsigned char> raw(g.size()*bytes);ggml_backend_tensor_get(y,raw.data(),0,raw.size());std::vector<float> out(g.size());
    for(size_t i=0;i<out.size();++i){if(bytes==4)memcpy(&out[i],raw.data()+i*4,4);else{ggml_fp16_t h;memcpy(&h,raw.data()+i*2,2);out[i]=ggml_fp16_to_fp32(h);}}
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);return out;
}
static void metrics(const char *label,const std::vector<float>&ref,const std::vector<float>&out,const std::vector<float>&g) {
    double sq=0,en=0,ma=0,mr=0;size_t neq=0,nf=0;double near=0,negative=0,positive=0;
    for(size_t i=0;i<ref.size();++i){if(!std::isfinite(ref[i]) || !std::isfinite(out[i])){nf+=!(std::isnan(ref[i])&&std::isnan(out[i])) && ref[i]!=out[i];continue;}
        double d=std::abs(double(out[i])-ref[i]);sq+=d*d;en+=double(ref[i])*ref[i];ma=std::max(ma,d);
        if(std::abs(ref[i])>1e-6)mr=std::max(mr,d/std::abs(ref[i]));
        neq+=memcmp(&ref[i],&out[i],4)!=0;
        if(std::abs(g[i])<1e-3)near=std::max(near,d);
        if(g[i]<-20)negative=std::max(negative,d);
        if(g[i]>20)positive=std::max(positive,d);
    }
    printf("GLU_NUMERICS %s elements=%zu nmse=%.12g max_abs=%.12g max_rel_ref_gt_1e_6=%.12g different=%zu nonfinite_mismatch=%zu near_zero_abs=%.12g large_negative_abs=%.12g large_positive_abs=%.12g\n",label,ref.size(),sq/std::max(en,1e-30),ma,mr,neq,nf,near,negative,positive);
    GGML_ASSERT(nf==0);
}
static std::vector<unsigned char> device_bytes(const void*p,size_t n){std::vector<unsigned char>v(n);kunlun::check(kunlun::copy(v.data(),p,n,kunlun::device_to_host));kunlun::check(kunlun::wait());return v;}
static void quant_compare(int pd,int w,const std::vector<float>&a,const std::vector<float>&b) {
    int rows=a.size()/w;kunlun::Buffer da(pd,a.size()*4),db(pd,b.size()*4);
    kunlun::check(kunlun::copy(da.data(),a.data(),a.size()*4,kunlun::host_to_device));kunlun::check(kunlun::copy(db.data(),b.data(),b.size()*4,kunlun::host_to_device));kunlun::check(kunlun::wait());
    kunlun::CommonQ4Workspace wa,wb;kunlun::CommonQ4ActivationLease la,lb;
    GGML_ASSERT(kunlun::common_q4_prepare(wa,pd,da.data(),w,rows,kunlun::InputType::f32,1,la));
    GGML_ASSERT(kunlun::common_q4_prepare(wb,pd,db.data(),w,rows,kunlun::InputType::f32,2,lb));
    GGML_ASSERT(la.chunks()==lb.chunks());size_t nq=0,ns=0,dq=0,ds=0;
    for(size_t i=0;i<la.chunks();++i){auto x=la.chunk(i),y=lb.chunk(i);auto aq=device_bytes(x.values,x.values_bytes),bq=device_bytes(y.values,y.values_bytes),as=device_bytes(x.scales,x.scales_bytes),bs=device_bytes(y.scales,y.scales_bytes);
        for(size_t j=0;j<aq.size();++j)dq+=aq[j]!=bq[j];
        for(size_t j=0;j<as.size();j+=4)ds+=memcmp(as.data()+j,bs.data()+j,4)!=0;
        nq+=aq.size();ns+=as.size()/4;
    }
    printf("GLU_QUANT_COMPARE width=%d rows=%d components=%d quant_bytes=%zu differing_bytes=%zu scale_floats=%zu differing_scales=%zu\n",w,rows,la.chunk(0).components,nq,dq,ns,ds);
}
int main(int argc,char**argv){
    GGML_ASSERT(argc==2 || argc==6 || argc==7);int pd=atoi(argv[1]);GGML_ASSERT(pd==0 || pd==1);setenv("GGML_KUNLUN_SWIGLU","1",1);
    ggml_backend_load_all();char name[32];snprintf(name,sizeof(name),"KUNLUN%d",pd);auto *gpu=ggml_backend_dev_by_name(name),*cpu=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);GGML_ASSERT(cpu&&gpu);
    int w=11008,bytes=4;std::vector<float>g,u;
    if(argc>=6){w=atoi(argv[2]);bytes=atoi(argv[3]);GGML_ASSERT(w>0 && w<=65536 && (bytes==2 || bytes==4));g=read(argv[4]);u=read(argv[5]);GGML_ASSERT(g.size()==u.size() && g.size()%w==0);}
    else{g.resize((size_t)w*8);u.resize(g.size());for(size_t i=0;i<g.size();++i){g[i]=((int)(i*31%4097)-2048)/128.f;u[i]=((int)(i*17%1025)-512)/128.f;}const float cases[]={-200,-100,-90,-80,-40,-20,-1,-1e-8f,-0.f,0,1e-8f,1,20,80,200};for(size_t i=0;i<g.size();i+=17)g[i]=cases[(i/17)%15];}
    GGML_ASSERT(bytes==2||bytes==4);if(bytes==2)for(size_t i=0;i<g.size();++i){g[i]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(g[i]));u[i]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(u[i]));}
    double cms,xms;auto c=run(cpu,g,u,w,bytes,cms),x=run(gpu,g,u,w,bytes,xms);std::vector<float>s(g.size()),d(g.size());
    for(size_t i=0;i<g.size();++i){s[i]=(g[i]/(1.f+std::exp(-g[i])))*u[i];d[i]=(float)((double(g[i])/(1.+std::exp(-double(g[i]))))*double(u[i]));if(bytes==2){s[i]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(s[i]));d[i]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(d[i]));}}
    if(argc==7){auto captured=read(argv[6]);GGML_ASSERT(captured.size()==c.size() && !memcmp(captured.data(),c.data(),c.size()*4));printf("GLU_CAPTURE CPU_replay_bitwise=1 elements=%zu\n",c.size());}
    metrics("CPU_SIMD_vs_XPU",c,x,g);metrics("scalar_libm_vs_XPU",s,x,g);metrics("F64_vs_CPU_SIMD",d,c,g);metrics("F64_vs_XPU",d,x,g);
    printf("GLU_TIME width=%d rows=%zu bytes=%d CPU_ms=%.9g XPU_ms=%.9g repetitions=3\n",w,g.size()/w,bytes,cms,xms);
    if(bytes==4)quant_compare(pd,w,c,x);
    return 0;
}

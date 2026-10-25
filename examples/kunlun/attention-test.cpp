#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static float value(int i){return ((i*31%1009)-504)/503.f;}
static void put(std::vector<unsigned char>&raw,size_t at,ggml_type type,float v){if(type==GGML_TYPE_F32)std::memcpy(raw.data()+at,&v,4);else{auto h=ggml_fp32_to_fp16(v);std::memcpy(raw.data()+at,&h,2);}}
static std::vector<std::vector<float>> evaluate(ggml_backend_dev_t device,int d,int s,int m,int hk,int hq,ggml_type type,bool rope) {
 auto backend=ggml_backend_dev_init(device,nullptr);GGML_ASSERT(backend);auto *ctx=ggml_init({4*1024*1024,nullptr,true});GGML_ASSERT(ctx);
 int bs=ggml_type_size(type);
 auto *ks=ggml_new_tensor_2d(ctx,type,d*hk,s+17),*vs=ggml_new_tensor_2d(ctx,type,s+17,d*hk),*qs=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,d*hq,m);
 auto *k=ggml_view_4d(ctx,ks,d,s,hk,1,ks->nb[1],d*bs,ggml_nbytes(ks),0);
 auto *v=ggml_view_4d(ctx,vs,s,d,hk,1,vs->nb[1],vs->nb[1]*d,ggml_nbytes(vs),0);
 auto *q0=ggml_reshape_3d(ctx,qs,d,hq,m);ggml_tensor *rp=nullptr,*positions=nullptr;
 if(rope){positions=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,m);rp=ggml_rope_ext(ctx,q0,positions,nullptr,d,0,2048,10000,1,0,1,32,1);q0=rp;}
 auto *q=ggml_permute(ctx,q0,0,2,1,3);
 auto *mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,s,m);
 auto *kq=ggml_mul_mat(ctx,k,q);GGML_ASSERT(ggml_prec_set_acc(kq,GGML_PREC_F32));
 auto *sm=ggml_soft_max_ext(ctx,kq,mask,1/std::sqrt((float)d),0);
 auto *out=ggml_mul_mat(ctx,v,sm),*cp=ggml_cont_2d(ctx,ggml_permute(ctx,out,0,2,1,3),d*hq,m);
 auto *graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,cp);
 for(int i=0;i<ggml_graph_n_nodes(graph);++i)GGML_ASSERT(ggml_backend_dev_supports_op(device,ggml_graph_node(graph,i)));
 auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);GGML_ASSERT(buffer);
 std::vector<unsigned char>kr(ggml_nbytes(ks)),vr(ggml_nbytes(vs)),qr(ggml_nbytes(qs)),mr(ggml_nbytes(mask));
 for(int h=0;h<hk;++h)for(int j=0;j<s;++j)for(int z=0;z<d;++z)put(kr,j*ks->nb[1]+(h*d+z)*bs,type,value(z+j*3+h*7)*.5f);
 for(int h=0;h<hk;++h)for(int j=0;j<d;++j)for(int z=0;z<s;++z)put(vr,(h*d+j)*vs->nb[1]+z*bs,type,value(z+j*7+h*3));
 for(int j=0;j<m;++j)for(int h=0;h<hq;++h)for(int z=0;z<d;++z)put(qr,j*qs->nb[1]+(h*d+z)*4,GGML_TYPE_F32,value(z+j*11+h*5)*.5f);
 for(int j=0;j<m;++j)for(int z=0;z<s;++z)put(mr,j*mask->nb[1]+z*4,GGML_TYPE_F32,z<=7+j?0:-INFINITY);
 for(auto pair:{std::make_pair(ks,&kr),std::make_pair(vs,&vr),std::make_pair(qs,&qr),std::make_pair(mask,&mr)})ggml_backend_tensor_set(pair.first,pair.second->data(),0,pair.second->size());
 if(positions){std::vector<int32_t>pos(m);for(int i=0;i<m;++i)pos[i]=123+i;ggml_backend_tensor_set(positions,pos.data(),0,m*4);}
 GGML_ASSERT(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
 std::vector<std::vector<float>> result;
 for(auto *t:{kq,sm,out,cp}){result.emplace_back(ggml_nelements(t));ggml_backend_tensor_get(t,result.back().data(),0,result.back().size()*4);}
 if(rp){result.emplace_back(ggml_nelements(rp));ggml_backend_tensor_get(rp,result.back().data(),0,result.back().size()*4);}
 // Attention must not mutate the resident cache, including inactive padding.
 for(auto pair:{std::make_pair(ks,&kr),std::make_pair(vs,&vr)}) {
  std::vector<unsigned char> got(pair.second->size());
  ggml_backend_tensor_get(pair.first,got.data(),0,got.size());
  GGML_ASSERT(got==*pair.second);
 }
 ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);return result;
}
// A 2D projection with a viewed activation remains the existing dense path.
static std::vector<float> linear_view(ggml_backend_dev_t device,ggml_type type,const char *attention) {
 setenv("GGML_KUNLUN_ATTENTION",attention,1);
 auto backend=ggml_backend_dev_init(device,nullptr);GGML_ASSERT(backend);
 auto *ctx=ggml_init({128*1024,nullptr,true});
 auto *w=ggml_new_tensor_2d(ctx,type,128,32);
 auto *storage=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,128,4);
 auto *x=ggml_view_2d(ctx,storage,128,3,storage->nb[1],storage->nb[1]);
 auto *y=ggml_mul_mat(ctx,w,x);GGML_ASSERT(ggml_backend_dev_supports_op(device,y));
 auto *graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
 auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);GGML_ASSERT(buffer);
 std::vector<unsigned char> wr(ggml_nbytes(w));std::vector<float> xr(512);
 for(int i=0;i<4096;++i)put(wr,i*ggml_type_size(type),type,value(i));
 for(int i=0;i<512;++i)xr[i]=value(i*3);
 ggml_backend_tensor_set(w,wr.data(),0,wr.size());ggml_backend_tensor_set(storage,xr.data(),0,xr.size()*4);
 GGML_ASSERT(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
 std::vector<float> out(96);ggml_backend_tensor_get(y,out.data(),0,out.size()*4);
 ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);return out;
}
int main(int argc,char **argv){
 int pd=argc>1?std::atoi(argv[1]):0;GGML_ASSERT(pd==0 || pd==1);ggml_backend_load_all();char name[32];std::snprintf(name,sizeof(name),"KUNLUN%d",pd);
 auto *gpu=ggml_backend_dev_by_name(name),*cpu=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);GGML_ASSERT(gpu&&cpu);
 for(auto type:{GGML_TYPE_F16,GGML_TYPE_F32})for(int test=0;test<4;++test){
  int d=test==3?128:64,s=test==2?513:257,m=test==0?1:test==1?7:test==2?9:3,hk=test==3?32:4,hq=32;bool rope=test==1;
  auto a=evaluate(cpu,d,s,m,hk,hq,type,rope),b=evaluate(gpu,d,s,m,hk,hq,type,rope);
  for(size_t op=0;op<a.size();++op){double sq=0,en=0,max=0;for(size_t i=0;i<a[op].size();++i){GGML_ASSERT(std::isfinite(b[op][i]));double delta=b[op][i]-a[op][i];sq+=delta*delta;en+=double(a[op][i])*a[op][i];max=std::max(max,std::abs(delta));}
   double nmse=sq/std::max(en,1e-30);std::printf("GGML_ATTENTION pd=%d type=%s test=%d stage=%zu nmse=%.12g max_abs=%.12g\n",pd,ggml_type_name(type),test,op,nmse,max);std::fflush(stdout);GGML_ASSERT(nmse<1e-6);
  }
 }
 for(auto type:{GGML_TYPE_F16,GGML_TYPE_F32}) {
  auto legacy=linear_view(gpu,type,"0"),enabled=linear_view(gpu,type,"1");
  GGML_ASSERT(legacy==enabled);
 }
 std::printf("GGML_ATTENTION regular 2D viewed-input projection unchanged PASS\n");
 auto *ctx=ggml_init({128*1024,nullptr,true});auto *a=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,64,257,4,1),*b=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,64,3,32,1);
 auto *mm=ggml_mul_mat(ctx,a,b);GGML_ASSERT(ggml_backend_dev_supports_op(gpu,mm));a->nb[0]=4;GGML_ASSERT(!ggml_backend_dev_supports_op(gpu,mm));a->nb[0]=2;
 auto *soft=ggml_soft_max_ext(ctx,mm,nullptr,.125f,0);GGML_ASSERT(ggml_backend_dev_supports_op(gpu,soft));ggml_soft_max_add_sinks(soft,ggml_new_tensor_1d(ctx,GGML_TYPE_F32,32));GGML_ASSERT(!ggml_backend_dev_supports_op(gpu,soft));
 setenv("GGML_KUNLUN_ATTENTION","tg",1);GGML_ASSERT(!ggml_backend_dev_supports_op(gpu,mm));
 auto *single=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,64,1,32,1);GGML_ASSERT(ggml_backend_dev_supports_op(gpu,ggml_mul_mat(ctx,a,single)));setenv("GGML_KUNLUN_ATTENTION","1",1);
 setenv("GGML_KUNLUN_ATTENTION","0",1);GGML_ASSERT(!ggml_backend_dev_supports_op(gpu,mm));setenv("GGML_KUNLUN_ATTENTION","1",1);
 // Whole-chain policy for a PP ubatch larger than the validated matrix bound.
 auto *largeq=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,64,512,32,1);
 auto *largemm=ggml_mul_mat(ctx,a,largeq);
 auto *largesm=ggml_soft_max_ext(ctx,largemm,nullptr,.125f,0);
 auto *largecp=ggml_cont_2d(ctx,ggml_permute(ctx,largemm,0,2,1,3),257*32,512);
 auto *rx=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,64,32,512);
 auto *pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,512);
 auto *rp=ggml_rope_ext(ctx,rx,pos,nullptr,64,0,2048,10000,1,0,1,32,1);
 for(auto *t:{largemm,largesm,largecp,rp})GGML_ASSERT(!ggml_backend_dev_supports_op(gpu,t));
 ggml_free(ctx);std::printf("GGML_ATTENTION unsupported stride/sinks/disabled/large query CPU placement, read-only KV PASS\n");
}

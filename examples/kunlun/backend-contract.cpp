#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <initializer_list>
#include <cstring>
#include <limits>

int main() {
    ggml_backend_load_all();
    auto * device=ggml_backend_dev_by_name("KUNLUN0");
    GGML_ASSERT(device);
    auto * context=ggml_init({128*1024,nullptr,true});
    GGML_ASSERT(context);
    for (int width : {16384,32768}) {
        auto * input=ggml_new_tensor_1d(context,GGML_TYPE_F32,width);
        ggml_tensor * nodes[]={ggml_add(context,input,input),ggml_mul(context,input,input),
            ggml_silu(context,input),ggml_scale(context,input,.5f),ggml_rms_norm(context,input,1e-5f)};
        for (auto * node : nodes) {
            bool supported=ggml_backend_dev_supports_op(device,node);
            GGML_ASSERT(supported == (width <= 16384));
            std::printf("operation=%s width=%d placement=%s PASS\n",ggml_op_name(node->op),width,supported ? "XPU" : "CPU");
        }
    }
    auto * input=ggml_new_tensor_1d(context,GGML_TYPE_F32,4096);
    for (float epsilon : {0.0f,-1.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        auto * node=ggml_rms_norm(context,input,1e-5f);
        std::memcpy(node->op_params,&epsilon,4);
        GGML_ASSERT(!ggml_backend_dev_supports_op(device,node));
        std::printf("operation=RMS_NORM epsilon=%g placement=CPU PASS\n",epsilon);
    }
    auto * rows=ggml_new_tensor_2d(context,GGML_TYPE_F32,4,2);
    auto * indices=ggml_new_tensor_1d(context,GGML_TYPE_I64,2);
    auto * cache=ggml_new_tensor_2d(context,GGML_TYPE_F16,4,4);
    auto * update=ggml_set_rows(context,cache,rows,indices);
    GGML_ASSERT(ggml_backend_dev_supports_op(device,update));
    size_t saved=update->nb[1];update->nb[1]=2;
    GGML_ASSERT(!ggml_backend_dev_supports_op(device,update));update->nb[1]=saved;
    saved=rows->nb[0];rows->nb[0]=2;
    GGML_ASSERT(!ggml_backend_dev_supports_op(device,update));rows->nb[0]=saved;
    std::printf("operation=SET_ROWS overlapping/misaligned layout placement=CPU PASS\n");
    ggml_free(context);
}

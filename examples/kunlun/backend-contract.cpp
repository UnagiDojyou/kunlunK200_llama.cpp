#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <initializer_list>

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
    ggml_free(context);
}

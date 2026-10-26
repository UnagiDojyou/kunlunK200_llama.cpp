bool attention_enabled() {
    const char *mode=std::getenv("GGML_KUNLUN_ATTENTION");return mode && (!std::strcmp(mode,"1") || !std::strcmp(mode,"tg"));
}
bool attention_rope_enabled() {
    const char *mode=std::getenv("GGML_KUNLUN_ROPE");return attention_enabled() && (!mode || !std::strcmp(mode,"1"));
}
bool attention_matrix_candidate(const ggml_tensor *t) {
    const auto *a=t->src[0],*b=t->src[1];
    return attention_enabled() && t->op==GGML_OP_MUL_MAT && a && b &&
        (a->type==GGML_TYPE_F16 || a->type==GGML_TYPE_F32) &&
        (a->view_src || !ggml_is_matrix(a) || !ggml_is_matrix(b));
}
bool attention_layout(const ggml_tensor *t,kunlun::Tensor4Layout &p) {
    if(!t || (t->type!=GGML_TYPE_F32 && t->type!=GGML_TYPE_F16))return false;
    for(int d=0;d<4;++d) {
        if(t->ne[d]<=0 || t->ne[d]>INT_MAX || t->nb[d]>INT_MAX)return false;
        p.ne[d]=t->ne[d];p.nb[d]=t->nb[d];
    }
    size_t bytes=ggml_type_size(t->type);
    if(t->view_offs%bytes || (t->data && (uintptr_t)t->data%bytes))return false;
    p.type=t->type==GGML_TYPE_F16?kunlun::InputType::f16:kunlun::InputType::f32;
    return !*kunlun::tensor4_reason(p);
}
bool flash_enabled() {
    const char *mode=std::getenv("GGML_KUNLUN_FLASH");return attention_enabled() && mode && !std::strcmp(mode,"1");
}
const char * flash_parameters(const ggml_tensor *t,kunlun::Flash4Layout &p) {
    if(!flash_enabled())return "flash disabled";
    if(t->src[4])return "flash sinks";
    if(!attention_layout(t->src[0],p.q)||!attention_layout(t->src[1],p.k)||!attention_layout(t->src[2],p.v)||!attention_layout(t,p.y))return "flash tensor layout/type";
    float cap;std::memcpy(&cap,(const char*)t->op_params+8,4);if(cap!=0)return "flash soft cap";
    p.has_mask=t->src[3]!=nullptr;if(p.has_mask&&!attention_layout(t->src[3],p.mask))return "flash mask";
    std::memcpy(&p.scale,t->op_params,4);std::memcpy(&p.max_bias,(const char*)t->op_params+4,4);
    return kunlun::flash_attention_reason(p);
}
void flash_compute(backend_context *ctx,ggml_tensor *t) {
    kunlun::Flash4Layout p{};GGML_ASSERT(!*flash_parameters(t,p));
    for(auto *v:{t,t->src[0],t->src[1],t->src[2],t->src[3]})if(v){auto *buffer=buffer_of(v);GGML_ASSERT(buffer&&!buffer->type->split&&buffer->type->id==ctx->id);}
    auto *q=t->src[0],*k=t->src[1],*v=t->src[2],*mask=t->src[3];
    bool ok=kunlun::flash_attention(ctx->attention,ctx->id,t->data,ggml_nbytes(t),q->data,ggml_nbytes(q),k->data,ggml_nbytes(k),v->data,ggml_nbytes(v),mask?mask->data:nullptr,mask?ggml_nbytes(mask):0,p);
    if(!ok)GGML_ABORT("KUNLUN Flash pointer/layout contract failed on PD%d",ctx->id);
    ++ctx->attention_phases["FLASH_ATTN_EXT"];
}
const char * attention_parameters(const ggml_tensor *t,kunlun::Matmul4Layout &p) {
    if(!attention_layout(t->src[0],p.a) || !attention_layout(t->src[1],p.b) || !attention_layout(t,p.y))return "matrix tensor layout/type";
    p.round_b_to_f16=true;return kunlun::attention_matmul_reason(p);
}
const char * attention_parameters(const ggml_tensor *t,kunlun::Softmax4Layout &p) {
    if(t->src[2])return "softmax sinks";
    if(!attention_layout(t->src[0],p.x) || !attention_layout(t,p.y))return "softmax tensor layout/type";
    p.has_mask=t->src[1]!=nullptr;
    if(p.has_mask && !attention_layout(t->src[1],p.mask))return "mask tensor layout/type";
    std::memcpy(&p.scale,t->op_params,4);std::memcpy(&p.max_bias,(const char*)t->op_params+4,4);
    return kunlun::attention_softmax_reason(p);
}
const char * attention_parameters(const ggml_tensor *t,kunlun::Rope4Layout &p) {
    const auto *pos=t->src[1];
    if(t->src[2])return "rope frequency tensor";
    if(!pos || pos->type!=GGML_TYPE_I32 || pos->view_offs%4 || (pos->data && (uintptr_t)pos->data%4) || !ggml_is_contiguous(pos) || pos->ne[0]<t->ne[2] || ggml_nbytes(pos)>INT_MAX)return "rope positions";
    if(!attention_layout(t->src[0],p.x) || !attention_layout(t,p.y))return "rope tensor layout/type";
    float ext;std::memcpy(&ext,t->op_params+7,4);if(ext!=0)return "rope YaRN";
    p.n_dims=t->op_params[1];p.mode=t->op_params[2];p.n_offs=t->op_params[15];
    std::memcpy(&p.freq_base,t->op_params+5,4);std::memcpy(&p.freq_scale,t->op_params+6,4);std::memcpy(&p.attn_factor,t->op_params+8,4);
    return kunlun::attention_rope_reason(p);
}
const char * attention_copy_reason(const ggml_tensor *t,kunlun::Tensor4Layout &p) {
    if(!attention_layout(t->src[0],p) || t->type!=t->src[0]->type || !ggml_is_contiguous(t) || ggml_nelements(t)!=ggml_nelements(t->src[0]) || ggml_nbytes(t)>INT_MAX)return "CONT tensor layout/type";
    return "";
}
const char * attention_reason(const ggml_tensor *t) {
    if(!attention_enabled())return "disabled";
    // Keep all stages of an unsupported large PP query on CPU. Accepting only
    // Softmax/CONT here would introduce new host/device round trips.
    int64_t queries=t->op==GGML_OP_MUL_MAT?t->src[1]->ne[1]:
        t->op==GGML_OP_ROPE?t->ne[2]:t->op==GGML_OP_SOFT_MAX?t->ne[1]:ggml_nrows(t);
    if(queries>256)return "query count exceeds Attention chain limit";
    if(!std::strcmp(std::getenv("GGML_KUNLUN_ATTENTION"),"tg") && queries>1)
        return "PP CPU policy";
    if(t->op==GGML_OP_MUL_MAT){kunlun::Matmul4Layout p{};return attention_parameters(t,p);}
    if(t->op==GGML_OP_SOFT_MAX){kunlun::Softmax4Layout p{};return attention_parameters(t,p);}
    if(t->op==GGML_OP_ROPE){if(!attention_rope_enabled())return "rope disabled";kunlun::Rope4Layout p{};return attention_parameters(t,p);}
    if(t->op==GGML_OP_CONT){kunlun::Tensor4Layout p{};return attention_copy_reason(t,p);}
    return "operation";
}
void attention_compute(backend_context *ctx,ggml_tensor *t) {
    GGML_ASSERT(!*attention_reason(t));
    for(const auto *v:{(const ggml_tensor*)t,(const ggml_tensor*)t->src[0],(const ggml_tensor*)t->src[1],(const ggml_tensor*)t->src[2]})if(v) {
        auto *buffer=buffer_of(v);GGML_ASSERT(buffer && !buffer->type->split && buffer->type->id==ctx->id);
    }
    const auto *a=t->src[0],*b=t->src[1];bool ok=false;
    if(t->op==GGML_OP_MUL_MAT) {
        kunlun::Matmul4Layout p{};GGML_ASSERT(!*attention_parameters(t,p));
        ok=kunlun::attention_matmul(ctx->attention,ctx->id,t->data,ggml_nbytes(t),a->data,ggml_nbytes(a),b->data,ggml_nbytes(b),p);
    } else if(t->op==GGML_OP_SOFT_MAX) {
        kunlun::Softmax4Layout p{};GGML_ASSERT(!*attention_parameters(t,p));
        ok=kunlun::attention_softmax(ctx->attention,ctx->id,t->data,ggml_nbytes(t),a->data,ggml_nbytes(a),b?b->data:nullptr,b?ggml_nbytes(b):0,p);
    } else if(t->op==GGML_OP_ROPE) {
        kunlun::Rope4Layout p{};GGML_ASSERT(!*attention_parameters(t,p));
        ok=kunlun::attention_rope(ctx->attention,ctx->id,t->data,ggml_nbytes(t),a->data,ggml_nbytes(a),(const int32_t*)b->data,ggml_nbytes(b),p);
    } else {
        kunlun::Tensor4Layout p{};GGML_ASSERT(!*attention_copy_reason(t,p));
        ok=kunlun::attention_contiguous(ctx->attention,ctx->id,t->data,ggml_nbytes(t),a->data,ggml_nbytes(a),p);
    }
    if(!ok)GGML_ABORT("KUNLUN Attention %s %s pointer/layout contract failed on PD%d",t->name,ggml_op_name(t->op),ctx->id);
    const char *phase=std::getenv("K200_PROFILE_PHASE");
    std::string op=t->op==GGML_OP_MUL_MAT?(std::strstr(t->name,"kqv")?"KQV":"KQ"):ggml_op_name(t->op);
    ++ctx->attention_phases[std::string(phase?phase:"other")+"|"+op];
    std::string shape=op+"|"+std::to_string(t->ne[0])+"x"+std::to_string(t->ne[1])+"x"+std::to_string(t->ne[2]);
    if(!ctx->attention_shapes[shape]++)std::fprintf(stderr,"KUNLUN Attention PD%d op=%s shape=%s device_resident=1\n",ctx->id,op.c_str(),shape.c_str());
}

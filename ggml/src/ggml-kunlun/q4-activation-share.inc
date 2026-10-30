// Graph-local ownership of explicitly prepared activation snapshots.
// The tensor object + graph/group generation owns identity; data addresses only
// validate the footprint and detect writes. No lease survives graph_compute.
bool activation_share_enabled() {
    const char * s=std::getenv("GGML_KUNLUN_Q4_ACT_SHARE");
    return common_enabled() && (!s || !std::strcmp(s,"1"));
}
bool activation_candidate(backend_context * c, const ggml_tensor * t) {
    if (t->op!=GGML_OP_MUL_MAT || !t->src[0] || !t->src[1]) return false;
    const auto * w=t->src[0], * x=t->src[1];
    auto * p=(tensor_parts *)w->extra;
    auto * wb=buffer_of(w), * xb=buffer_of(x);
    return wb && p && p->common && xb && !xb->type->split && xb->type->id==c->id &&
        x->type==GGML_TYPE_F32 && ggml_is_matrix(x) && ggml_is_contiguous(x) && x->data &&
        x->ne[0]==w->ne[0] && x->ne[0]>0 && x->ne[0]<=16384 && x->ne[1]>0 &&
        x->ne[1]<=INT_MAX/x->ne[0]/4 && ggml_nbytes(x)<=(size_t)INT_MAX;
}
void activation_clear(backend_context * c) {
    auto & a=c->activation;
    for (auto & l:a.lease) l.reset();
    a.input=nullptr;a.data=nullptr;a.buffer=nullptr;a.bytes=0;a.use=false;
    a.peer_generation=0;a.peer_pointer=nullptr;a.peer_extent=0;
}
bool activation_same(const ActivationShare & a,const ggml_tensor * x) {
    if(a.input!=x || a.data!=x->data || a.buffer!=(x->view_src?x->view_src->buffer:x->buffer) || a.type!=x->type) return false;
    for(int d=0;d<4;++d) if(a.ne[d]!=x->ne[d] || a.nb[d]!=x->nb[d]) return false;
    return a.bytes==ggml_nbytes(x);
}
void activation_begin(backend_context * c,ggml_tensor * t) {
    K200_DIAG_SCOPE(activation_diag,"xpu.activation","select",t->name);
    auto & a=c->activation;a.use=false;
    if(!a.enabled || !activation_candidate(c,t)) return;
    const auto * x=t->src[1];auto found=a.remaining.find(x);
    if(found==a.remaining.end()) return;
    bool same=activation_same(a,x);
    if(!same && found->second<2) return;
    if(!same) {
        activation_clear(c);GGML_ASSERT(++a.generation!=0);
        a.input=x;a.data=x->data;a.buffer=x->view_src?x->view_src->buffer:x->buffer;
        a.type=x->type;a.bytes=ggml_nbytes(x);
        for(int d=0;d<4;++d){a.ne[d]=x->ne[d];a.nb[d]=x->nb[d];}
        ++a.groups;
    }
    a.use=true;
}
void activation_end(backend_context * c,const ggml_tensor * t) {
    K200_DIAG_SCOPE(activation_diag,"xpu.activation","release",t->name);
    auto & a=c->activation;if(!a.enabled || !t->src[1])return;
    auto i=a.remaining.find(t->src[1]);
    if(i!=a.remaining.end() && activation_candidate(c,t)) {
        GGML_ASSERT(i->second>0);
        if(!--i->second && a.input==t->src[1])activation_clear(c);
    }
    a.use=false;
}
void activation_write(backend_context * c,const ggml_tensor * t) {
    K200_DIAG_SCOPE(activation_diag,"xpu.activation","write_check",t->name);
    auto & a=c->activation;if(!a.enabled || !a.input)return;
    if(t->op==GGML_OP_NONE || t->op==GGML_OP_VIEW || t->op==GGML_OP_RESHAPE ||
       t->op==GGML_OP_PERMUTE || t->op==GGML_OP_TRANSPOSE)return;
    auto * b=buffer_of(t);
    if(!b || b->type->split || b->type->id!=c->id)return;
    uintptr_t p=(uintptr_t)t->data,q=(uintptr_t)a.data;size_t n=ggml_nbytes(t);
    if(p+n<p || q+a.bytes<q || (p<q+a.bytes && q<p+n)) {
        ++a.write_invalidations;activation_clear(c);
    }
}
bool activation_ready(backend_context * c,int id,int k,int m) {
    auto & a=c->activation;
    return a.use && a.lease[id].matches(id,k,m,kunlun::InputType::f32,a.generation);
}
bool activation_matmul(backend_context * c,int id,void * y,const kunlun::CommonQ4Weight & w,const void * x,int k,int m) {
    auto & a=c->activation;if(!a.use)return false;
    auto declined=[&](const char * stage) {
        if(!a.fallback[id]++)std::fprintf(stderr,
            "KUNLUN activation fallback PD%d: stage=%s K=%d N=%d M=%d generation=%llu; using ordinary common Q4\n",
            id,stage,k,w.info().n,m,(unsigned long long)a.generation);
    };
    bool reused=activation_ready(c,id,k,m);
    if(!reused) {
        if(!kunlun::common_q4_prepare(c->common[id],id,x,k,m,kunlun::InputType::f32,a.generation,a.lease[id])) {
            declined("prepare-contract-or-capacity");return false;
        }
        ++a.prepares[id];
    }
    if(kunlun::common_q4_matmul_shared(c->common[id],id,y,w,a.lease[id],a.generation)) {
        if(reused)++a.hits[id];
        return true;
    }
    declined("prepared-contract-or-L3");a.lease[id].reset();return false;
}
struct ActivationGraphScope {
    backend_context * c;
#ifdef K200_DIAGNOSTIC_HOOKS
    kunlun::CommonQ4ActivationStatistics before[2]{};
    std::string phase;
#endif
    explicit ActivationGraphScope(backend_context * owner,ggml_cgraph * graph):c(owner) {
        K200_DIAG_SCOPE(activation_diag,"xpu.activation","plan","");
        auto & a=c->activation;
#ifdef K200_DIAGNOSTIC_HOOKS
        if(a.profile) {
            const char * p=std::getenv("K200_PROFILE_PHASE");phase=p?p:"other";
            for(int id=0;id<2;++id)before[id]=c->common[id].activation_statistics();
        }
#endif
        if(!a.enabled)return;
        activation_clear(c);a.remaining.clear();
        ++a.graphs;
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto * t=ggml_graph_node(graph,i);
            if(activation_candidate(c,t))++a.remaining[t->src[1]];
        }
    }
    ~ActivationGraphScope(){
#ifdef K200_DIAGNOSTIC_HOOKS
        if(c->activation.profile)for(int id=0;id<2;++id) {
            auto after=c->common[id].activation_statistics();auto & total=c->activation.phase_quant[id][phase];
            total.prepare_calls+=after.prepare_calls-before[id].prepare_calls;
            total.quantized_chunks+=after.quantized_chunks-before[id].quantized_chunks;
            total.quant_kernel_launches+=after.quant_kernel_launches-before[id].quant_kernel_launches;
            total.prepared_matmuls+=after.prepared_matmuls-before[id].prepared_matmuls;
            total.quantization_ns+=after.quantization_ns-before[id].quantization_ns;
            total.preparation_ns+=after.preparation_ns-before[id].preparation_ns;
            total.hbm_activation_bytes=std::max(total.hbm_activation_bytes,after.hbm_activation_bytes);
        }
#endif
        K200_DIAG_SCOPE(activation_diag,"xpu.activation","graph_release","");
        if(c->activation.enabled){activation_clear(c);c->activation.remaining.clear();}
    }
};
struct ActivationMatmulScope {
    backend_context * c;const ggml_tensor * t;
    ActivationMatmulScope(backend_context * owner,ggml_tensor * node):c(owner),t(node){activation_begin(c,node);}
    ~ActivationMatmulScope(){activation_end(c,t);}
};

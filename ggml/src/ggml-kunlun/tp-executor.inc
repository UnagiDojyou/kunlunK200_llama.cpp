// Called only by Meta. All borrowed graph and tensor pointers expire on return.
bool tp_enabled() {
    const char * s = kunlun_default_on("GGML_KUNLUN_TENSOR_PARALLEL");
    return s && !std::strcmp(s,"1");
}
bool tp_trace() {
    const char * s = std::getenv("GGML_KUNLUN_TP_TRACE");
    return s && !std::strcmp(s,"1");
}
bool tp_group_enabled() {
    const char * s=std::getenv("GGML_KUNLUN_TP_GROUP");
    return !s || std::strcmp(s,"0");
}
bool tp_use_child_probe(ggml_backend_dev_t) { return tp_enabled(); }

bool tp_span(const ggml_tensor * t, int id) {
    if (!t || !t->data || ggml_nelements(t)<=0) return false;
    auto * b=buffer_of(t);
    if (!b || b->type->split || b->type->id!=id) return false;
    uintptr_t base=(uintptr_t)b->base, p=(uintptr_t)t->data;
    size_t bytes=ggml_nbytes(t);
    if (!(p>=base && p-base<=b->size && bytes<=b->size-(p-base))) return false;
    if (t->view_src) {
        const auto * root=t;
        while (root->view_src) root=root->view_src;
        if ((root->type==GGML_TYPE_Q4_0 || root->type==GGML_TYPE_Q6_K) && root->extra) return false;
        uintptr_t r=(uintptr_t)root->data;
        if (p<r || p-r>ggml_nbytes(root) || bytes>ggml_nbytes(root)-(p-r)) return false;
    }
    return true;
}
bool tp_f32_binding(const ggml_tensor * t) {
    if ((uintptr_t)t->data%4) return false;
    const auto * root=t;
    while (root->view_src) root=root->view_src;
    return root->type==GGML_TYPE_F32 && !root->extra;
}
bool tp_plain_f32(const ggml_tensor * t,int id) {
    return tp_span(t,id) && t->type==GGML_TYPE_F32 && ggml_is_contiguous(t) &&
        ggml_nelements(t)<=INT_MAX/4 && t->ne[0]>0 && t->ne[0]<=16384 && tp_f32_binding(t);
}

// No device addresses or allocation handles are cached. Meta owns the graph objects for this UID.
struct TPPreflightLayout {
    const ggml_tensor * tensor;
    const ggml_tensor * view;
    size_t view_offset;
    ggml_type type;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];

    explicit TPPreflightLayout(const ggml_tensor * t) : tensor(t),view(t->view_src),view_offset(t->view_offs),type(t->type) {
        std::memcpy(ne,t->ne,sizeof(ne));
        std::memcpy(nb,t->nb,sizeof(nb));
    }
    bool matches() const {
        return tensor->view_src==view && tensor->view_offs==view_offset && tensor->type==type &&
            std::memcmp(ne,tensor->ne,sizeof(ne))==0 && std::memcmp(nb,tensor->nb,sizeof(nb))==0;
    }
};

struct TPPreflightNode {
    const ggml_tensor * logical;
    ggml_backend_meta_split_axis axis;
    const ggml_tensor * child;
    const ggml_tensor * src[GGML_MAX_SRC];
    ggml_op op;
    int32_t parameter;
    bool first_output;
};

struct TPPreflightGroup {
    const ggml_cgraph * other = nullptr;
    std::vector<TPPreflightLayout> layouts[2];
    std::vector<TPPreflightNode> nodes[2];
};

struct TPGroupExecutor {
    ggml_backend_t backends[2];
    backend_context * contexts[2];
    std::unique_ptr<kunlun::Buffer> receive;
    int sum_sync_supported = -1;
    std::unique_ptr<kunlun::Buffer> gate_up[2];
    size_t gate_glu_calls[2]={}, gate_glu_fused_calls=0;
    kunlun::CommonQ4ActivationLease leases[2];
    const ggml_tensor * inputs[2]={};
    uint64_t invocation=0;
    size_t groups=0, parallel_nodes=0, serial_nodes=0, reductions=0, bytes=0;
    bool pending[2]={};
    uint64_t preflight_generation=0;
    bool preflight_ffn=false, preflight_attention=false;
    std::map<const ggml_cgraph *,TPPreflightGroup> preflight;
    size_t preflight_hits=0, preflight_misses=0, preflight_rebuilds=0;
    size_t preflight_bytes=0, preflight_peak_bytes=0;
    bool drain() {
        bool ok=true;
        for (int d=0;d<2;++d) if (pending[d]) {
            select(d);
            const int status=kunlun::wait();
            // A failed wait is not proof of completion. Keep the pending bit.
            if (status) ok=false; else pending[d]=false;
        }
        return ok;
    }
};

// Allocation handles are borrowed for this invocation, never cached by address.
struct TPShardDescriptor {
    const ggml_backend_meta_node_descriptor & meta;
    uint64_t graph_generation, allocation_epoch;
    backend_context * owners[2];
    bool complete[2]={true,true};
    bool requires_sum() const { return meta.split_state->axis==GGML_BACKEND_SPLIT_AXIS_PARTIAL; }
};

bool tp_split_valid(ggml_backend_meta_split_axis axis) {
    return (axis>=GGML_BACKEND_SPLIT_AXIS_0 && axis<=GGML_BACKEND_SPLIT_AXIS_3) ||
           axis==GGML_BACKEND_SPLIT_AXIS_MIRRORED || axis==GGML_BACKEND_SPLIT_AXIS_PARTIAL;
}

void * tp_comm_init(ggml_backend_t * backends,size_t n) {
    if (!tp_enabled() || n!=2 || backends[0]==backends[1]) return nullptr;
    for (int d=0;d<2;++d) if (ggml_backend_get_device(backends[d])!=&devices[d]) return nullptr;
    auto * g=new TPGroupExecutor{};
    for (int d=0;d<2;++d) {
        g->backends[d]=backends[d];
        g->contexts[d]=(backend_context *)backends[d]->context;
    }
    GGML_LOG_INFO("KUNLUN TP: 2PD group, F32 SUM, normal HBM, single host thread\n");
    return g;
}
void tp_comm_free(void * opaque) {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
    auto * g=(TPGroupExecutor *)opaque;
    // Fail closed: do not release workspace after an uncompleted device error.
    if (!g->drain()) GGML_ABORT("KUNLUN TP: drain failed; retain buffers and stop process");
    GGML_LOG_INFO("KUNLUN TP totals: groups=%zu parallel=%zu serial=%zu reductions=%zu p2p=%zu bytes=%zu\n",
        g->groups,g->parallel_nodes,g->serial_nodes,g->reductions,2*g->reductions,g->bytes);
    if (g->preflight_misses) GGML_LOG_INFO("KUNLUN TP preflight cache: hits=%zu misses=%zu rebuilds=%zu peak_bytes=%zu\n",
        g->preflight_hits,g->preflight_misses,g->preflight_rebuilds,g->preflight_peak_bytes);
    if (g->gate_glu_calls[0] || g->gate_glu_calls[1]) {
        GGML_LOG_INFO("KUNLUN TP Gate/Up SwiGLU: M1=%zu M128=%zu fused_sum_glu=%zu\n",
            g->gate_glu_calls[0],g->gate_glu_calls[1],g->gate_glu_fused_calls);
    }
    delete g;
}
bool tp_m1_sum_input(const ggml_tensor * t, int64_t shard_k, const char * suffix) {
    const auto * w=t->src[0];
    if (t->op != GGML_OP_MUL_MAT || t->ne[0] != 4096 || ggml_nelements(t) != 4096 ||
        (uintptr_t)t->data % 256 || !w || w->type != GGML_TYPE_Q4_0 ||
        w->ne[0] != shard_k || w->ne[1] != 4096) {
        return false;
    }
    const size_t name_length = std::strlen(w->name);
    const size_t suffix_length = std::strlen(suffix);
    return name_length >= suffix_length && std::strcmp(w->name+name_length-suffix_length,suffix) == 0;
}

bool tp_sum_flag(const char * name) {
    const char * flag=kunlun_default_on(name);
    return flag && std::strcmp(flag,"1")==0;
}
void tp_m1_sum_inplace(TPGroupExecutor * g, ggml_tensor * t) {
    g->pending[0] = true;
    const int rc = kunlun::tp_wo_sum_submit(g->contexts[0]->ew, t->data, g->receive->data());
    if (!g->drain()) {
        GGML_ABORT("KUNLUN TP M1 SUM: failed to drain; retain buffers and stop");
    }
    // The original partials cannot be trusted after an attempted in-place ADD.
    if (rc) {
        GGML_ABORT("KUNLUN TP M1 SUM: launch failed after collective began");
    }
}

bool tp_allreduce_impl(void * opaque,ggml_tensor ** t,bool group_complete) {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
    auto * g=(TPGroupExecutor *)opaque;
    if (!g || !t || !tp_plain_f32(t[0],0) || !tp_plain_f32(t[1],1) || !ggml_are_same_shape(t[0],t[1])) {
        GGML_LOG_ERROR("KUNLUN TP SUM declined: require equal contiguous F32 normal HBM on PD0/PD1\n");
        return false;
    }
    if (!g->drain()) GGML_ABORT("KUNLUN TP SUM: producer drain failed");
    K200_DIAG_SCOPE(reduce_diag,"tp.reduce","SUM");
    // Direct users of the comm hook may have submitted work outside this group.
    if (!group_complete) {
        for (int d=0;d<2;++d) { select(d); check(kunlun::wait()); }
    }
    size_t bytes=ggml_nbytes(t[0]);
    if (!g->receive || g->receive->size()<bytes) g->receive.reset(new kunlun::Buffer(0,kunlun::aligned(bytes)));
    const bool wo = group_complete &&
        tp_m1_sum_input(t[0],2048,".attn_output.weight") && tp_m1_sum_input(t[1],2048,".attn_output.weight");
    const bool down = group_complete &&
        tp_m1_sum_input(t[0],5504,".ffn_down.weight") && tp_m1_sum_input(t[1],5504,".ffn_down.weight");
    const bool fast_copy = (wo && tp_sum_flag("GGML_KUNLUN_TP_WO_SUM_COPY")) ||
                           (down && tp_sum_flag("GGML_KUNLUN_TP_DOWN_SUM_COPY"));
    if (fast_copy && g->sum_sync_supported < 0) {
        g->sum_sync_supported = kunlun::tp_wo_sum_sync_available();
    }
    const auto transfer = fast_copy && g->sum_sync_supported == 1 ?
        kunlun::tp_wo_sum_sync_peer : kunlun::peer_completed;
    transfer(0,g->receive->data(),1,t[1]->data,bytes);
    const int elements=(int)ggml_nelements(t[0]), width=(int)t[0]->ne[0];
    const char * inplace=kunlun_default_on("GGML_KUNLUN_TP_SUM_INPLACE");
    if ((wo && tp_sum_flag("GGML_KUNLUN_TP_WO_SUM_ADD")) ||
        (down && tp_sum_flag("GGML_KUNLUN_TP_DOWN_SUM_ADD"))) {
        tp_m1_sum_inplace(g,t[0]);
    } else if (inplace && std::strcmp(inplace,"1")==0 && width==4096 && elements==4096 &&
        (uintptr_t)t[0]->data%64==0 && (uintptr_t)g->receive->data()%64==0) {
        kunlun::tp_sum_inplace(g->contexts[0]->ew,0,t[0]->data,g->receive->data(),elements,width);
    } else {
        kunlun::elementwise(g->contexts[0]->ew,0,t[0]->data,t[0]->data,g->receive->data(),
            elements,width,elements,kunlun::ElementwiseOp::add);
    }
    kunlun::kv_max_invalidate(0,t[0]->data,bytes);
    transfer(1,t[1]->data,0,t[0]->data,bytes);
    ++g->reductions;g->bytes+=2*bytes;
    if (tp_trace()) GGML_LOG_INFO("KUNLUN TP SUM: bytes=%zu copies=2 order=PD0+PD1 complete=1\n",bytes);
    return true;
}

bool tp_allreduce(void * opaque,ggml_tensor ** t) {
    return tp_allreduce_impl(opaque,t,false);
}

bool tp_metadata_op(const ggml_tensor * t) {
    return t->op==GGML_OP_NONE || t->op==GGML_OP_VIEW || t->op==GGML_OP_RESHAPE ||
           t->op==GGML_OP_PERMUTE || t->op==GGML_OP_TRANSPOSE;
}
bool tp_async_matmul(const ggml_tensor * t,int d) {
    return t->op==GGML_OP_MUL_MAT && tp_plain_f32(t,d) && tp_plain_f32(t->src[0],d) &&
           tp_plain_f32(t->src[1],d) && !t->src[0]->view_src &&
           supports_op(&devices[d],t) &&
           // The CL matmul has no input/output alias contract.
           ((uintptr_t)t->data+ggml_nbytes(t)<=(uintptr_t)t->src[0]->data ||
            (uintptr_t)t->src[0]->data+ggml_nbytes(t->src[0])<=(uintptr_t)t->data) &&
           ((uintptr_t)t->data+ggml_nbytes(t)<=(uintptr_t)t->src[1]->data ||
            (uintptr_t)t->src[1]->data+ggml_nbytes(t->src[1])<=(uintptr_t)t->data);
}

#include "tp-q4.inc"

bool tp_exact_or_disjoint(const ggml_tensor * out, const ggml_tensor * in) {
    const uintptr_t a = (uintptr_t)out->data, b = (uintptr_t)in->data;
    if (a == b) return ggml_nbytes(out) == ggml_nbytes(in);
    return a < b ? ggml_nbytes(out) <= b-a : ggml_nbytes(in) <= a-b;
}

bool tp_ew_flag(const char * name) {
    const char * value=kunlun_default_on(name);
    return value && std::strcmp(value,"1")==0;
}

bool tp_elementwise_layout(ggml_tensor ** t) {
    for (int d=0; d<2; ++d) {
        const auto * out=t[d], * a=out->src[0], * b=out->src[1] ? out->src[1] : a;
        if (!tp_plain_f32(out,d) || !tp_plain_f32(a,d) || !tp_plain_f32(b,d) ||
            !ggml_are_same_shape(out,a) || !tp_exact_or_disjoint(out,a) || !tp_exact_or_disjoint(out,b) ||
            (ggml_nelements(b) != ggml_nelements(out) && ggml_nelements(b) != out->ne[0])) return false;
    }
    return true;
}

int tp_elementwise_submit(TPGroupExecutor * g, ggml_tensor * out, int d,
                          kunlun::ElementwiseSubmitOptions options) {
    const auto op=out->op;
    K200_DIAG_SCOPE(submit_diag,"tp.ew","submit",ggml_op_name(op),0,d);
    const auto kind=op == GGML_OP_ADD ? kunlun::ElementwiseOp::add :
                    op == GGML_OP_MUL ? kunlun::ElementwiseOp::multiply : kunlun::ElementwiseOp::rms_norm;
    const auto * a=out->src[0], * b=out->src[1] ? out->src[1] : a;
    float parameter;
    std::memcpy(&parameter,out->op_params,sizeof(parameter));
    select(d);
    kunlun::kv_max_invalidate(d,out->data,ggml_nbytes(out));
    g->pending[d]=true;
    return (options.exact_alias || options.ordered_rms) ?
        kunlun::elementwise_submit_ex(g->contexts[d]->ew,d,out->data,a->data,b->data,
            ggml_nelements(out),out->ne[0],ggml_nelements(b),kind,parameter,options) :
        kunlun::elementwise_submit(g->contexts[d]->ew,d,out->data,a->data,b->data,
            ggml_nelements(out),out->ne[0],ggml_nelements(b),kind,parameter);
}

int tp_elementwise_complete(TPGroupExecutor * g, ggml_op op, int status) {
    for (int d=0; d<2; ++d) if (g->pending[d]) {
        K200_DIAG_SCOPE(complete_diag,"tp.ew","complete",ggml_op_name(op),0,d);
        select(d);
        const int wait_status=kunlun::wait();
        if (!wait_status) g->pending[d]=false;
        else status=wait_status;
    }
    if (!g->drain()) GGML_ABORT("KUNLUN TP: elementwise drain failed");
    return status ? -1 : 1;
}

int tp_elementwise_execute(TPGroupExecutor * g, ggml_tensor ** t, const TPShardDescriptor & shard) {
    const auto op=t[0]->op;
    if ((op != GGML_OP_ADD && op != GGML_OP_MUL && op != GGML_OP_RMS_NORM) || t[1]->op != op ||
        shard.meta.split_state->axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) return 0;
    if (!tp_ew_flag("GGML_KUNLUN_TP_ELEMENTWISE") || tp_serial()) return 0;
    const char * rms=std::getenv("GGML_KUNLUN_TP_RMS");
    if (op == GGML_OP_RMS_NORM && rms && std::strcmp(rms,"0")==0) return 0;
    const kunlun::ElementwiseSubmitOptions options{
        tp_ew_flag("GGML_KUNLUN_TP_EW_INPLACE"),tp_ew_flag("GGML_KUNLUN_TP_EW_RMS_QUEUE")};
    if (options.exact_alias || options.ordered_rms) {
        for (int d=0; d<2; ++d) for (int k=0; k<2; ++k) {
            const auto * in=t[d]->src[k];
            if (in && in->data && t[d]->data && in->data!=t[d]->data && !tp_exact_or_disjoint(t[d],in)) return -1;
        }
    }
    if (!tp_elementwise_layout(t)) return 0;
    const auto begin=tp_now();
    int status=0;
    for (int d=0; d<2 && !status; ++d) {
        status=tp_elementwise_submit(g,t[d],d,options);
    }
    if (tp_elementwise_complete(g,op,status)<0) return -1;
    ++g->parallel_nodes;
    tp_record(g,op == GGML_OP_ADD ? TP_SYNC : TP_NORM,begin);
    return 1;
}

void tp_invalidate_activation(TPGroupExecutor * g, const ggml_tensor * t, int d) {
    if (!g->inputs[d]) return;
    uintptr_t dst=(uintptr_t)t->data, src=(uintptr_t)g->inputs[d]->data;
    if (dst<src+ggml_nbytes(g->inputs[d]) && src<dst+ggml_nbytes(t)) {
        g->leases[d].reset();
        g->inputs[d]=nullptr;
    }
}

int tp_elementwise_chain(TPGroupExecutor * g, ggml_cgraph ** graphs,
                          const ggml_backend_meta_node_descriptor * desc, size_t i) {
    ggml_tensor * nodes[3][2];
    const ggml_op ops[3]={GGML_OP_ADD,GGML_OP_RMS_NORM,GGML_OP_MUL};
    for (int j=0; j<3; ++j) {
        if (desc[i+j].split_state->axis!=GGML_BACKEND_SPLIT_AXIS_MIRRORED) return 0;
        for (int d=0; d<2; ++d) {
            auto * t=ggml_graph_node(graphs[d],i+j);
            nodes[j][d]=t;
            if (t->op!=ops[j] || t->ne[0]!=4096 || ggml_nelements(t)!=4096 || t->view_src ||
                (j && t->src[0]!=nodes[j-1][d])) return 0;
            for (const ggml_tensor * in:{(const ggml_tensor *)t,(const ggml_tensor *)t->src[0],
                                         (const ggml_tensor *)t->src[1]}) {
                if (in && (uintptr_t)in->data%64) return 0;
            }
        }
        if (!tp_elementwise_layout(nodes[j])) return 0;
    }
    const auto begin=tp_now();
    K200_DIAG_SCOPE(chain_diag,"tp.ew.chain","ADD/RMS/MUL");
    // Keep every intermediate at its assigned address for graph outputs and other consumers.
    int status=0;
    for (int j=0; j<3 && !status; ++j) {
        K200_DIAG_SCOPE(node_diag,"tp.node",ggml_op_name(ops[j]),nodes[j][0]->name);
        for (int d=0; d<2 && !status; ++d) {
            tp_invalidate_activation(g,nodes[j][d],d);
            status=tp_elementwise_submit(g,nodes[j][d],d,{true,true});
        }
    }
    if (tp_elementwise_complete(g,GGML_OP_MUL,status)<0) return -1;
    g->parallel_nodes+=3;
    tp_record(g,TP_NORM,begin);
    return 1;
}

int tp_kv_update(TPGroupExecutor * g, ggml_tensor ** t, const TPShardDescriptor & shard) {
    if (t[0]->op!=GGML_OP_SET_ROWS || t[1]->op!=GGML_OP_SET_ROWS ||
        shard.meta.split_state->axis!=GGML_BACKEND_SPLIT_AXIS_0) return 0;
    const char * flag=kunlun_default_on("GGML_KUNLUN_TP_KV_UPDATE");
    const char * mode=std::getenv("GGML_KUNLUN_SET_ROWS");
    if (!flag || std::strcmp(flag,"1") || (mode && !std::strcmp(mode,"legacy")) || tp_serial()) return 0;
    const auto * logical=shard.meta.logical;
    const auto * index=logical->src[1];
    ggml_backend_meta_split_state split{};
    // A one-copy scheduler input is a mirrored Meta leaf without the INPUT flag.
    if (!index || index->op!=GGML_OP_NONE || index->view_src ||
        !ggml_is_contiguous(index) || !ggml_backend_meta_tensor_split_state(index,true,&split) ||
        split.axis!=GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
        (index->type!=GGML_TYPE_I32 && index->type!=GGML_TYPE_I64) ||
        index->ne[1]!=1 || index->ne[2]!=1 || index->ne[3]!=1 || index->ne[0]>128 || index->ne[0]<1) return 0;
    const int rows=index->ne[0];
    kunlun::KVRowUpdate jobs[2]{};
    for (int d=0;d<2;++d) {
        auto * out=t[d];
        const auto * src=out->src[0], * idx=out->src[1], * cache=out->src[2];
        if (!cache || !src || !idx || idx!=ggml_backend_meta_tensor_child(index,d) ||
            cache!=ggml_backend_meta_tensor_child(logical->src[2],d) ||
            !tp_span(out,d) || !tp_span(src,d) || !tp_span(idx,d) || !tp_span(cache,d) ||
            !ggml_are_same_shape(out,cache) || out->data!=cache->data || !ggml_is_contiguous(out) ||
            !ggml_is_contiguous(src) || !ggml_is_contiguous(idx) || !ggml_are_same_shape(idx,index) || idx->type!=index->type ||
            (src->type!=GGML_TYPE_F32 && src->type!=GGML_TYPE_F16) || !set_rows_layout_supported(out,src,idx) ||
            src->ne[0]!=out->ne[0] || src->ne[1]!=rows || src->ne[2]!=1 || src->ne[3]!=1 ||
            out->ne[2]!=1 || out->ne[3]!=1 || out->ne[1]>INT_MAX ||
            out->data==src->data || out->data==idx->data || !tp_exact_or_disjoint(out,src) || !tp_exact_or_disjoint(out,idx)) return 0;
        auto & j=jobs[d];
        j={&g->contexts[d]->kv,out->data,ggml_nbytes(out),src->data,ggml_nbytes(src),idx->data,ggml_nbytes(idx),{}};
        auto & p=j.layout;
        p.width=src->ne[0];p.rows=rows;p.dim2=1;p.dim3=1;
        for (int i=0;i<4;++i) p.source_stride[i]=src->nb[i];
        for (int i=0;i<3;++i) {p.destination_stride[i]=out->nb[i+1];p.index_stride[i]=idx->nb[i];}
        p.index_extent[0]=1;p.index_extent[1]=1;
        p.source_bytes=ggml_type_size(src->type);p.index_bytes=ggml_type_size(idx->type);p.destination_rows=out->ne[1];
    }
    if (!ggml_are_same_shape(t[0],t[1]) || !ggml_are_same_shape(t[0]->src[0],t[1]->src[0])) return 0;
    const auto begin=tp_now();
    std::array<int64_t,128> targets{};
    {
        K200_DIAG_SCOPE(index_diag,"tp.kv","index",index->name);
        // Meta writes both child inputs from the same logical input. The nodes borrow it for this invocation only.
        std::array<unsigned char,128*sizeof(int64_t)> values;
        ggml_backend_tensor_get(t[0]->src[1],values.data(),0,ggml_nbytes(index));
        for (int i=0;i<rows;++i) {
            if (index->type==GGML_TYPE_I64) std::memcpy(&targets[i],values.data()+i*sizeof(int64_t),sizeof(int64_t));
            else { int32_t v;std::memcpy(&v,values.data()+i*sizeof(v),sizeof(v));targets[i]=v; }
            if (targets[i]<0 || targets[i]>=jobs[0].layout.destination_rows) {
                GGML_LOG_ERROR("KUNLUN TP: SET_ROWS index out of range before submission\n");
                return -1;
            }
        }
        std::sort(targets.begin(),targets.begin()+rows);
        if (std::adjacent_find(targets.begin(),targets.begin()+rows)!=targets.begin()+rows) return 0;
    }
    {
        K200_DIAG_SCOPE(write_diag,"tp.kv","pair",index->name);
        if (!kunlun::kv_set_rows_pair(jobs,targets.data(),rows)) return -1;
    }
    ++g->parallel_nodes;
    tp_record(g,TP_KV,begin);
    return 1;
}

bool tp_first_output(const ggml_backend_meta_node_descriptor & desc) {
    if (!desc.logical || !desc.children || !desc.children[0] || !desc.children[1] || !desc.split_state ||
        !(desc.logical->flags & GGML_TENSOR_FLAG_META_FIRST_OUTPUT) || desc.logical->op != GGML_OP_MUL_MAT ||
        !desc.logical->src[0] || desc.logical->src[0]->type != GGML_TYPE_Q6_K) return false;
    const auto & split = *desc.split_state;
    return split.axis == GGML_BACKEND_SPLIT_AXIS_0 && split.n_segments == 1 && split.nr[0] == 1 &&
        split.ne[0] == desc.logical->ne[0] && split.ne[1] == 0 &&
        ggml_are_same_shape(desc.children[0], desc.logical) && desc.children[1]->ne[0] == 0 &&
        !(desc.children[1]->flags & GGML_TENSOR_FLAG_COMPUTE);
}

int tp_q6_output_pair(TPGroupExecutor * g, const ggml_backend_meta_node_descriptor & desc) {
    const char * enabled = kunlun_default_on("GGML_KUNLUN_TP_OUTPUT_PARALLEL");
    if (!enabled || std::strcmp(enabled,"1") != 0 ||
            !(desc.logical->flags & GGML_TENSOR_FLAG_META_Q6_PAIR)) return 0;
    auto * y = desc.children[0];
    const auto * w = desc.logical->src[0];
    const auto * x = desc.logical->src[1];
    if (!w || !x || w->ne[0] != 4096 || w->ne[1] != 32000 ||
            !tp_span(y,0) || y->type != GGML_TYPE_F32 || y->view_src ||
            y->ne[0] != 32000 || ggml_nelements(y) != 32000 ||
            !ggml_is_contiguous(y) || (uintptr_t)y->data%256) return 0;
    ggml_backend_meta_split_state state;
    for (const auto * src : {w,x}) {
        if (!ggml_backend_meta_tensor_split_state(src,true,&state) ||
                state.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) return 0;
    }
    const ggml_tensor * input[2];
    const kunlun::Q6KWeight * weights[2];
    for (int pd=0; pd<2; ++pd) {
        input[pd] = ggml_backend_meta_tensor_child(x,pd);
        const auto * packed = ggml_backend_meta_tensor_child(w,pd);
        if (!tp_plain_f32(input[pd],pd) || input[pd]->ne[0] != 4096 ||
                ggml_nelements(input[pd]) != 4096 || !tp_span(packed,pd) ||
                !use_q6(packed) || !packed->extra) return 0;
        const auto * parts = static_cast<const tensor_parts *>(packed->extra);
        if (!parts->q6 || !parts->q6_weight[pd] || parts->q6_weight[pd]->data() != packed->data) return 0;
        weights[pd] = parts->q6_weight[pd].get();
        const auto * buffer = buffer_of(packed);
        const size_t offset = (uintptr_t)packed->data - (uintptr_t)buffer->base;
        if (weights[pd]->info().payload_bytes > buffer->size - offset) return 0;
    }
    const char * flag = std::getenv("GGML_KUNLUN_TP_OUTPUT_DIAG");
    const bool diagnostic = flag && std::strcmp(flag,"1") == 0;
    kunlun::Q6KPairTiming timing{};
    K200_DIAG_SCOPE(head_diag,"tp.head","2PD",y->name,64000);
    const int status = kunlun::q6_k_matmul_pair(g->contexts[0]->q6[0],g->contexts[1]->q6[1],y->data,
                                              *weights[0],*weights[1],input[0]->data,input[1]->data,
                                              diagnostic ? &timing : nullptr);
    if (status == 1) {
        ++g->parallel_nodes;
        for (int pd=0;pd<2;++pd) ++g->contexts[pd]->q6_calls[pd];
        if (diagnostic) {
            GGML_LOG_INFO("KUNLUN_Q6_PAIR prepare=%llu quant0=%llu quant1=%llu sd0=%llu sd1=%llu "
                          "submit0=%llu submit1=%llu complete0=%llu complete1=%llu wait0=%llu wait1=%llu peer=%llu\n",
                    (unsigned long long)(timing.prepared_ns-timing.begin_ns),
                    (unsigned long long)timing.quant_enqueue_ns[0],(unsigned long long)timing.quant_enqueue_ns[1],
                    (unsigned long long)timing.sdnn_enqueue_ns[0],(unsigned long long)timing.sdnn_enqueue_ns[1],
                    (unsigned long long)timing.submit_ns[0],(unsigned long long)timing.submit_ns[1],
                    (unsigned long long)timing.complete_ns[0],(unsigned long long)timing.complete_ns[1],
                    (unsigned long long)timing.wait_ns[0],(unsigned long long)timing.wait_ns[1],
                    (unsigned long long)(timing.peer_end_ns-timing.peer_begin_ns));
        }
    }
    return status;
}

void tp_save_preflight(TPGroupExecutor * g,ggml_cgraph ** graphs,
        const ggml_backend_meta_node_descriptor * descriptors,size_t n) {
    auto & entry=g->preflight[graphs[0]];
    entry.other=graphs[1];
    g->preflight_bytes+=sizeof(entry);
    for (int d=0;d<2;++d) {
        std::vector<const ggml_tensor *> distinct;
        entry.nodes[d].reserve(n);
        for (size_t i=0;i<n;++i) {
            const auto & desc=descriptors[i];
            const auto * t=ggml_graph_node(graphs[d],i);
            const bool empty=d==1 && tp_first_output(desc);
            TPPreflightNode node{desc.logical,desc.split_state->axis,t,{},t->op,t->op_params[0],empty};
            std::copy_n(t->src,GGML_MAX_SRC,node.src);
            entry.nodes[d].push_back(node);
            if (empty) continue;
            distinct.push_back(t);
            for (const auto * src:t->src) if (src) distinct.push_back(src);
        }
        std::sort(distinct.begin(),distinct.end(),std::less<const ggml_tensor *>{});
        distinct.erase(std::unique(distinct.begin(),distinct.end()),distinct.end());
        entry.layouts[d].reserve(distinct.size());
        for (const auto * t:distinct) entry.layouts[d].emplace_back(t);
        g->preflight_bytes+=entry.nodes[d].capacity()*sizeof(TPPreflightNode)+
                            entry.layouts[d].capacity()*sizeof(TPPreflightLayout);
    }
    g->preflight_peak_bytes=std::max(g->preflight_peak_bytes,g->preflight_bytes);
}

bool tp_recheck_preflight(const TPPreflightGroup & entry,ggml_cgraph ** graphs,
        const ggml_backend_meta_node_descriptor * descriptors,size_t n,bool ffn,bool attention) {
    if (graphs[1]!=entry.other || entry.nodes[0].size()!=n) return false;
    for (int d=0;d<2;++d) {
        if (!graphs[d] || (size_t)ggml_graph_n_nodes(graphs[d])!=n) return false;
        for (size_t i=0;i<n;++i) {
            const auto & desc=descriptors[i];
            const auto & cached=entry.nodes[d][i];
            const auto * t=ggml_graph_node(graphs[d],i);
            if (!desc.children || !desc.split_state || desc.logical!=cached.logical || t!=cached.child ||
                desc.children[d]!=t || desc.split_state->axis!=cached.axis || t->op!=cached.op ||
                std::memcmp(t->src,cached.src,sizeof(cached.src))!=0) return false;
            if (cached.first_output && !tp_first_output(desc)) return false;
        }
        for (const auto & layout:entry.layouts[d]) {
            if (!layout.matches() || !tp_span(layout.tensor,d)) return false;
        }
        for (const auto & cached:entry.nodes[d]) {
            const auto * t=cached.child;
            if (cached.first_output || tp_metadata_op(t)) continue;
            if (!(t->flags&GGML_TENSOR_FLAG_COMPUTE)) return false;
            if (cached.axis==GGML_BACKEND_SPLIT_AXIS_PARTIAL && !tp_f32_binding(t)) return false;
            const bool q4=t->op==GGML_OP_MUL_MAT && t->src[0]->type==GGML_TYPE_Q4_0;
            if (q4 && ffn && (!tp_f32_binding(t) || !tp_f32_binding(t->src[1]) || !tp_q4_binding(t->src[0],d))) return false;
            if (attention && (t->op==GGML_OP_ROPE || t->op==GGML_OP_FLASH_ATTN_EXT) && !tp_attention_supported(t)) return false;
            switch (t->op) {
                case GGML_OP_RMS_NORM:
                    if (cached.parameter!=t->op_params[0]) return false;
                    break;
                case GGML_OP_ADD: case GGML_OP_MUL:
                    break;
                case GGML_OP_SET_ROWS:
                    if ((uintptr_t)t->data%2 || (uintptr_t)t->src[0]->data%ggml_type_size(t->src[0]->type) ||
                        (uintptr_t)t->src[1]->data%ggml_type_size(t->src[1]->type)) return false;
                    break;
                case GGML_OP_FLASH_ATTN_EXT:
                    if (attention) break; // tp_attention_supported includes flash_parameters.
                    if (!supports_op(&devices[d],t)) return false;
                    break;
                default:
                    if (!q4 && !supports_op(&devices[d],t)) return false;
                    break;
            }
        }
    }
    return true;
}

ggml_status tp_compute_group(void * opaque,ggml_cgraph ** graphs,
        const ggml_backend_meta_node_descriptor * descriptors,size_t n,uint64_t graph_generation) {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
    auto * g=(TPGroupExecutor *)opaque;
    if (!g || !graphs || !descriptors) return GGML_STATUS_FAILED;
    if (!g->drain()) GGML_ABORT("KUNLUN TP: entry drain failed; buffers cannot be reused");
    ++g->invocation;
    for(int d=0;d<2;++d)g->inputs[d]=nullptr;
    K200_DIAG_SCOPE(group_diag,"tp.group","group");
    // Validate the whole subgraph before any submission.
    {
    K200_DIAG_SCOPE(preflight_diag,"tp.preflight","validation");
    const char * cache_flag=kunlun_default_on("GGML_KUNLUN_TP_PREFLIGHT_CACHE");
    const bool cache_on=graph_generation && cache_flag && std::strcmp(cache_flag,"1")==0;
    const bool ffn=tp_ffn_enabled(), attention=tp_attention_enabled();
    if (!cache_on && !g->preflight.empty()) {
        g->preflight.clear();
        g->preflight_generation=0;
        g->preflight_bytes=0;
    }
    if (cache_on && (g->preflight_generation!=graph_generation || g->preflight_ffn!=ffn || g->preflight_attention!=attention)) {
        g->preflight.clear();
        g->preflight_generation=graph_generation;
        g->preflight_ffn=ffn;
        g->preflight_attention=attention;
        g->preflight_bytes=0;
        ++g->preflight_rebuilds;
    }
    auto cached=cache_on ? g->preflight.find(graphs[0]) : g->preflight.end();
    if (cached!=g->preflight.end()) {
        K200_DIAG_SCOPE(cache_diag,"tp.cache","hit");
        if (!tp_recheck_preflight(cached->second,graphs,descriptors,n,ffn,attention)) {
            GGML_LOG_ERROR("KUNLUN TP: cached preflight rejected a changed layout or invalid binding\n");
            return GGML_STATUS_FAILED;
        }
        ++g->preflight_hits;
    } else {
    K200_DIAG_SCOPE(cache_diag,"tp.cache","full");
    for (int d=0;d<2;++d) {
        if (!graphs[d] || (size_t)ggml_graph_n_nodes(graphs[d])!=n) return GGML_STATUS_FAILED;
        for (size_t i=0;i<n;++i) {
            const auto * t=ggml_graph_node(graphs[d],i);
            const auto & desc=descriptors[i];
            if (d == 1 && desc.children && desc.children[d] == t && tp_first_output(desc)) continue;
            if (!desc.logical || !desc.split_state || !desc.children || desc.children[d]!=t ||
                !tp_split_valid(desc.split_state->axis) || !tp_span(t,d) ||
                (!tp_metadata_op(t) && (!(t->flags&GGML_TENSOR_FLAG_COMPUTE) || !supports_op(&devices[d],t)))) {
                GGML_LOG_ERROR("KUNLUN TP: invalid child/placement node=%zu PD%d\n",i,d);
                return GGML_STATUS_FAILED;
            }
            if (desc.split_state->axis==GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                (!tp_plain_f32(t,d) || !ggml_are_same_shape(desc.children[0],desc.children[1]))) {
                GGML_LOG_ERROR("KUNLUN TP: unsupported PARTIAL collective layout node=%zu PD%d\n",i,d);
                return GGML_STATUS_FAILED;
            }
            if(ffn && t->op==GGML_OP_MUL_MAT && t->src[0]->type==GGML_TYPE_Q4_0 && !tp_q4_tensor(t,d)) {
                GGML_LOG_ERROR("KUNLUN TP FFN preflight: unsupported Q4 shape/physical buffer\n");return GGML_STATUS_FAILED;
            }
            if(attention && (t->op==GGML_OP_ROPE || t->op==GGML_OP_FLASH_ATTN_EXT) && !tp_attention_supported(t)) {
                GGML_LOG_ERROR("KUNLUN TP Attention preflight: unsupported Head/position/layout on PD%d\n",d);return GGML_STATUS_FAILED;
            }
            for (const auto * src:t->src) if (src && !tp_span(src,d)) {
                GGML_LOG_ERROR("KUNLUN TP: foreign or stale input node=%zu PD%d\n",i,d);
                return GGML_STATUS_FAILED;
            }
        }
    }
    if (cache_on) {
        tp_save_preflight(g,graphs,descriptors,n);
        ++g->preflight_misses;
    }
    }
    }
    ++g->groups;
    const char * gate_glu=kunlun_default_on("GGML_KUNLUN_TP_GATE_GLU");
    const bool fuse_gate_glu=gate_glu && std::strcmp(gate_glu,"1")==0 && !tp_serial() && tp_ffn_enabled();
    const char * rms_mode=std::getenv("KUNLUN_RMS_NORM");
    const char * tp_rms=std::getenv("GGML_KUNLUN_TP_RMS");
    const bool ew_chain=tp_ew_flag("GGML_KUNLUN_TP_EW_CHAIN") &&
        tp_ew_flag("GGML_KUNLUN_TP_EW_INPLACE") && tp_ew_flag("GGML_KUNLUN_TP_EW_RMS_QUEUE") &&
        tp_ew_flag("GGML_KUNLUN_TP_ELEMENTWISE") && !tp_serial() &&
        !(rms_mode && std::strcmp(rms_mode,"legacy")==0) && !(tp_rms && std::strcmp(tp_rms,"0")==0);
    for (size_t i=0;i<n;++i) {
        if (ew_chain && i+2<n && ggml_graph_node(graphs[0],i)->op==GGML_OP_ADD) {
            const int chained=tp_elementwise_chain(g,graphs,descriptors,i);
            if (chained<0) return GGML_STATUS_FAILED;
            if (chained) {
                i+=2;
                continue;
            }
        }
        if (fuse_gate_glu && i+2<n && tp_gate_glu_execute(g,graphs,descriptors,i)) {
            i+=2;
            continue;
        }
        TPShardDescriptor shard{descriptors[i],graph_generation,g->invocation,{g->contexts[0],g->contexts[1]}};
        ggml_tensor * t[2]={ggml_graph_node(graphs[0],i),ggml_graph_node(graphs[1],i)};
        if (tp_metadata_op(t[0]) && tp_metadata_op(t[1])) continue;
        for (int d=0; d<2; ++d) tp_invalidate_activation(g,t[d],d);
        K200_DIAG_SCOPE(node_diag,"tp.node",ggml_op_name(t[0]->op),
                t[0]->op==GGML_OP_MUL_MAT?t[0]->src[0]->name:t[0]->name);
        if (tp_first_output(descriptors[i])) {
            const int output_pair = tp_q6_output_pair(g,descriptors[i]);
            if (output_pair < 0) {
                GGML_LOG_ERROR("KUNLUN TP: Q6_K output submission or collection failed\n");
                return GGML_STATUS_FAILED;
            }
            if (output_pair > 0) continue;
            K200_DIAG_SCOPE(head_diag,"tp.head","PD0",t[0]->name,0,0);
            ggml_cgraph one{};
            one.n_nodes=1;one.nodes=graphs[0]->nodes+i;
            one.use_counts=graphs[0]->use_counts;
            one.visited_hash_set=graphs[0]->visited_hash_set;
            one.order=graphs[0]->order;
            const auto status=ggml_backend_graph_compute(g->backends[0],&one);
            if (status != GGML_STATUS_SUCCESS) return status;
            ++g->serial_nodes;
            continue;
        }
        if(i+1<n){ggml_tensor * next[2]={ggml_graph_node(graphs[0],i+1),ggml_graph_node(graphs[1],i+1)};int pair=tp_pair_execute(g,t,next);if(pair<0)return GGML_STATUS_FAILED;if(pair){++i;continue;}}
        int handled=tp_q4_execute(g,t);
        if(!handled)handled=tp_glu_execute(g,t);
        if(!handled)handled=tp_attention_execute(g,t);
        if(!handled)handled=tp_elementwise_execute(g,t,shard);
        if(!handled)handled=tp_kv_update(g,t,shard);
        if(handled<0){GGML_LOG_ERROR("KUNLUN TP FFN: unsupported/declined node=%zu op=%s\n",i,ggml_op_name(t[0]->op));return GGML_STATUS_FAILED;}
        if(handled)continue;
        if (tp_async_matmul(t[0],0) && tp_async_matmul(t[1],1)) {
            int64_t submit[2],complete[2];
            auto stamp=[]() { return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count(); };
            for (int d=0;d<2;++d) {
                auto * c=g->contexts[d];auto * a=t[d]->src[0];auto * b=t[d]->src[1];
                select(d);kunlun::kv_max_invalidate(d,t[d]->data,ggml_nbytes(t[d]));
                // Existing raw F32 matmul only submits. Kernels and scratch are PD-local.
                kunlun::matmul(c->mv[d],d,t[d]->data,a->data,b->data,a->ne[0],a->ne[1],b->ne[1],kunlun::WeightType::f32);
                g->pending[d]=true;shard.complete[d]=false;submit[d]=stamp();
            }
            for (int d=0;d<2;++d) {
                select(d);check(kunlun::wait());g->pending[d]=false;shard.complete[d]=true;complete[d]=stamp();
            }
            ++g->parallel_nodes;
            if (tp_trace()) GGML_LOG_INFO("KUNLUN TP window: graph=%llu invocation=%llu node=%zu submit0=%lld submit1=%lld complete0=%lld complete1=%lld\n",
                (unsigned long long)graph_generation,(unsigned long long)g->invocation,i,
                (long long)submit[0],(long long)submit[1],(long long)complete[0],(long long)complete[1]);
        } else {
            if (!g->drain()) GGML_ABORT("KUNLUN TP: drain before synchronous node failed");
            K200_DIAG_SCOPE(serial_diag,"tp.serial",ggml_op_name(t[0]->op),t[0]->name);
            const uint64_t sync_begin=tp_now();
            if (tp_trace()) GGML_LOG_INFO("KUNLUN TP serial: node=%zu op=%s\n",i,ggml_op_name(t[0]->op));
            for (int d=0;d<2;++d) {
                ggml_cgraph one{};
                one.n_nodes=1;one.nodes=graphs[d]->nodes+i;
                one.use_counts=graphs[d]->use_counts;
                one.visited_hash_set=graphs[d]->visited_hash_set;
                one.order=graphs[d]->order;
                ggml_status status=ggml_backend_graph_compute(g->backends[d],&one);
                if (status!=GGML_STATUS_SUCCESS) return status;
            }
            ++g->serial_nodes;
            if(tp_ffn_enabled())tp_record(g,t[0]->op==GGML_OP_SET_ROWS?TP_KV:(t[0]->op==GGML_OP_RMS_NORM || t[0]->op==GGML_OP_MUL?TP_NORM:TP_SYNC),sync_begin);
        }
        GGML_ASSERT(shard.complete[0] && shard.complete[1]);
    }
    if (!g->drain()) GGML_ABORT("KUNLUN TP: exit drain failed; buffers cannot be reused");
    return GGML_STATUS_SUCCESS;
}

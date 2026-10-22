#include "ggml-kunlun.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include <kunlun/k200.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <sys/mman.h>

namespace {
std::recursive_mutex & runtime_mutex = kunlun::runtime_mutex();
std::atomic<size_t> split_matmuls;
std::atomic<size_t> common_paths[2][8], common_uploads[2], common_upload_bytes[2], common_regions[2], common_region_bytes[2];
std::atomic<size_t> common_pack_ns[2], common_h2d_ns[2];
std::map<int,size_t> common_m[2];
std::map<std::string,std::array<size_t,8>> common_phase[2];
bool common_enabled() {
    const char * s=std::getenv("GGML_KUNLUN_COMMON_Q4");
    return s && std::strcmp(s,"1")==0;
}
void log_common_totals() {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
    if (!common_enabled() && !common_regions[0] && !common_regions[1]) return;
    for (int id=0;id<2;++id) {
        std::fprintf(stderr,"KUNLUN common totals PD%d: gemv=%zu combined=%zu wide=%zu rows64=%zu rows128=%zu combo2=%zu combo4=%zu exact_cl=%zu regions=%zu payload_bytes=%zu uploads=%zu upload_bytes=%zu pack_ms=%.3f upload_ms=%.3f\n",
            id,common_paths[id][0].load(),common_paths[id][1].load(),common_paths[id][2].load(),common_paths[id][3].load(),common_paths[id][4].load(),common_paths[id][5].load(),common_paths[id][6].load(),common_paths[id][7].load(),
            common_regions[id].load(),common_region_bytes[id].load(),common_uploads[id].load(),common_upload_bytes[id].load(),common_pack_ns[id].load()/1e6,common_h2d_ns[id].load()/1e6);
        for (auto pair : common_phase[id]) {
            std::fprintf(stderr,"KUNLUN common phase PD%d: phase=%s",id,pair.first.c_str());
            for(int i=0;i<8;++i)std::fprintf(stderr," p%d=%zu",i,pair.second[i]);
            std::fprintf(stderr,"\n");
        }
        for (auto pair : common_m[id]) std::fprintf(stderr,"KUNLUN common M PD%d: m=%d calls=%zu\n",id,pair.first,pair.second);
    }
}
std::atomic<size_t> native_uploads[2], native_upload_bytes[2], native_upload_ns[2];
std::atomic<size_t> native_sd_total[2], native_cl_total[2], native_cdnn_total[2], native_direct_total[2];
void log_native_totals() {
    const char * setting=std::getenv("GGML_KUNLUN_NATIVE_Q4");
    if (!common_enabled() && !std::getenv("GGML_KUNLUN_DIAGNOSTICS") && !(setting && std::strcmp(setting,"1")==0) && !(native_sd_total[0]+native_sd_total[1]+native_cl_total[0]+native_cl_total[1])) return;
    for (int id=0;id<2;++id) std::fprintf(stderr,
        "KUNLUN native totals PD%d: native_sd=%zu native_cl=%zu cdnn=%zu direct_q4=%zu native_uploads=%zu native_upload_bytes=%zu native_setup_ms=%.3f\n",
        id,native_sd_total[id].load(),native_cl_total[id].load(),native_cdnn_total[id].load(),native_direct_total[id].load(),
        native_uploads[id].load(),native_upload_bytes[id].load(),native_upload_ns[id].load()/1e6);
}
using kunlun::check;
using kunlun::select;
using kunlun::allocate;
using kunlun::release;
using kunlun::peer;
using kernel = kunlun::Kernel;
struct device_context { int id; std::string name; };
struct type_context { int id; bool split; double fraction; std::string name; };
struct tensor_parts {
    void * ptr[2] = {};
    size_t bytes[2] = {}, physical[2] = {};
    int rows[2] = {}, k = 0;
    bool packed = false, native = false, common = false;
    std::unique_ptr<kunlun::CommonQ4Weight> common_weight[2];
    std::vector<char> original;
};
struct buffer_context {
    type_context * type;
    void * base;
    size_t size;
    std::vector<std::unique_ptr<tensor_parts>> tensors;
};
struct backend_context {
    int id;
    kernel mv[2], ew, gather, kv;
    kunlun::NativeQ4Workspace native[2];
    kunlun::CommonQ4Workspace common[2];
    size_t i8_mm[2] = {}, q4_mm[2] = {};
    bool native_logged[2] = {}, fallback_logged[2] = {};
    void * split_x = nullptr;
    void * split_y = nullptr;
    void * local_y = nullptr;
    void * peer_y = nullptr;
    size_t split_x_size = 0, split_y_size = 0;
    size_t local_y_size = 0;
    size_t peer_y_size = 0;
};

ggml_backend_device devices[2];
ggml_backend_buffer_type normal_types[2];
type_context normal_contexts[2];
ggml_backend_buffer_i buffer_iface;
ggml_backend_buffer_type_i type_iface;

bool is_ours(ggml_backend_buffer_type_t bt) { return bt && bt->iface.get_name == type_iface.get_name; }
buffer_context * buffer_of(const ggml_tensor * t) {
    auto * buf = t->view_src ? t->view_src->buffer : t->buffer;
    return buf && is_ours(buf->buft) ? (buffer_context *)buf->context : nullptr;
}
bool use_common(const ggml_tensor * t) {
    return common_enabled() && t->type==GGML_TYPE_Q4_0 && ggml_is_matrix(t) && ggml_is_contiguous(t) && !t->view_src &&
        t->ne[0]%32==0 && t->ne[0]<=INT_MAX && t->ne[1]<=INT_MAX && kunlun::common_q4_supported(t->ne[0],t->ne[1]);
}
bool use_native(const ggml_tensor * t) {
    const char * setting=std::getenv("GGML_KUNLUN_NATIVE_Q4");
    return !use_common(t) && setting && std::strcmp(setting,"1") == 0 && t->type == GGML_TYPE_Q4_0 &&
        ggml_is_matrix(t) && ggml_is_contiguous(t) && !t->view_src &&
        t->ne[0] <= INT_MAX && t->ne[1] <= INT_MAX &&
        kunlun::native_q4_layout_supported(t->ne[0],t->ne[1]);
}
bool use_i8(const ggml_tensor * t) {
    const char * setting = std::getenv("GGML_KUNLUN_REQUANT_I8");
    return !use_common(t) && !use_native(t) && setting && std::strcmp(setting,"0") != 0 && t->type == GGML_TYPE_Q4_0 && ggml_is_matrix(t) && !t->view_src &&
        t->ne[0] > 0 && t->ne[0] <= 16384 && t->ne[0]%32 == 0 && t->ne[1] > 0 && t->ne[1] <= INT_MAX/t->ne[0];
}
size_t packed_size(int k, int n) { return kunlun::packed_i8_size(k,n); }
float * scales(tensor_parts * p, int id) { return kunlun::packed_i8_scales(p->ptr[id],p->k,p->rows[id]); }
float * maxima(tensor_parts * p, int id) { return kunlun::packed_i8_maxima(p->ptr[id],p->k,p->rows[id]); }

void repack(tensor_parts * p, size_t first, size_t last) {
    size_t row_bytes=ggml_row_size(GGML_TYPE_Q4_0,p->k), start=0;
    for (int id=0;id<2;++id) {
        size_t lo=std::max(first,start), hi=std::min(last,start+p->rows[id]);
        if (p->common) {
            if (hi > lo) {
                auto & w=*p->common_weight[id];auto before=w.statistics();
                w.upload(p->original.data()+start*row_bytes,p->bytes[id]);auto after=w.statistics();
                common_uploads[id]+=after.uploads-before.uploads;common_upload_bytes[id]+=after.upload_bytes-before.upload_bytes;
                common_pack_ns[id]+=(size_t)((after.pack_ms-before.pack_ms)*1e6);
                common_h2d_ns[id]+=(size_t)((after.upload_ms-before.upload_ms)*1e6);
            }
            start+=p->rows[id];continue;
        }
        if (p->native) {
            if (hi > lo) {
                auto begin=std::chrono::steady_clock::now();
                kunlun::upload_native_q4(id,p->ptr[id],p->original.data()+start*row_bytes,p->k,p->rows[id]);
                ++native_uploads[id]; native_upload_bytes[id]+=p->physical[id];
                native_upload_ns[id]+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
            }
            start+=p->rows[id]; continue;
        }
        // Bound staging memory while preserving the original Q4_0 bytes for GGML.
        for (size_t r=lo;r<hi;) {
            size_t count=std::min<size_t>(hi-r,64);
            std::vector<float> rows(count*p->k);
            for (size_t j=0;j<count;++j)
                ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float(p->original.data()+(r+j)*row_bytes,rows.data()+j*p->k,p->k);
            kunlun::upload_i8_rows(id,p->ptr[id],p->k,p->rows[id],r-start,count,rows.data());
            r+=count;
        }
        start+=p->rows[id];
    }
}

void transfer(ggml_backend_buffer_t buf, const ggml_tensor * t, void * data, size_t off, size_t size, bool put) {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
    GGML_ASSERT(off <= ggml_nbytes(t) && size <= ggml_nbytes(t) - off);
    auto * ctx = (buffer_context *)buf->context;
    const ggml_tensor * root = t;
    while (root->view_src) root=root->view_src;
    auto * packed = root->type == GGML_TYPE_Q4_0 ? (tensor_parts *)root->extra : nullptr;
    if (packed && (packed->packed || packed->native || packed->common)) {
        GGML_ASSERT((uintptr_t)t->data >= (uintptr_t)root->data);
        size_t start = (uintptr_t)t->data - (uintptr_t)root->data + off;
        GGML_ASSERT(start <= packed->original.size() && size <= packed->original.size()-start);
        if (put && size) {
            std::memcpy(packed->original.data()+start,data,size);
            size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0,packed->k);
            repack(packed,start/row_bytes,(start+size+row_bytes-1)/row_bytes);
        } else if (!put) std::memcpy(data,packed->original.data()+start,size);
        return;
    }
    if (!ctx->type->split) {
        select(ctx->type->id);
        void * addr = (char *)t->data + off;
        check(put ? kunlun::copy(addr, data, size, kunlun::host_to_device) : kunlun::copy(data, addr, size, kunlun::device_to_host));
        check(kunlun::wait()); return;
    }
    auto * parts = (tensor_parts *)root->extra;
    GGML_ASSERT(parts);
    off += (uintptr_t)t->data - (uintptr_t)root->data;
    GGML_ASSERT(off <= ggml_nbytes(root) && size <= ggml_nbytes(root)-off);
    size_t start = 0;
    for (int d = 0; d < 2; ++d) {
        size_t lo = std::max(off, start), hi = std::min(off + size, start + parts->bytes[d]);
        if (hi > lo) {
            select(d);
            void * addr = (char *)parts->ptr[d] + lo - start;
            void * host = (char *)data + lo - off;
            check(put ? kunlun::copy(addr, host, hi-lo, kunlun::host_to_device) : kunlun::copy(host, addr, hi-lo, kunlun::device_to_host));
            check(kunlun::wait());
        }
        start += parts->bytes[d];
    }
}

void clear_device(int id, void * ptr, size_t size, uint8_t value) {
    if (!size) return;
    select(id);
    std::vector<char> data(std::min<size_t>(size,1<<20),value);
    for (size_t offset=0;offset<size;offset+=data.size())
        check(kunlun::copy((char *)ptr+offset,data.data(),std::min(data.size(),size-offset),kunlun::host_to_device));
    check(kunlun::wait());
}

ggml_backend_buffer_t alloc_buffer(ggml_backend_buffer_type_t bt, size_t size) {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
    auto * tc = (type_context *)bt->context;
    void * base = tc->split ? mmap(nullptr, std::max<size_t>(size, 1), PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) : allocate(tc->id, size);
    if (!base || base == MAP_FAILED) return nullptr;
    auto * ctx = new buffer_context{tc, base, size, {}};
    return ggml_backend_buffer_init(bt, buffer_iface, ctx, size);
}

void launch_mv(backend_context * ctx, int id, void * y, const void * w, const void * x, int k, int n, int batch, int type, tensor_parts * packed = nullptr) {
    if (packed && packed->common) {
        GGML_ASSERT(packed->common_weight[id] && packed->common_weight[id]->data()==w);
        auto info=packed->common_weight[id]->info();
        GGML_ASSERT(type==2 && info.k==k && info.n==n && info.device==id);
        auto & workspace=ctx->common[id];auto before=workspace.statistics();
        kunlun::common_q4_matmul(workspace,id,y,*packed->common_weight[id],x,batch);
        auto after=workspace.statistics();
        const char * phase=std::getenv("K200_PROFILE_PHASE");
        auto & phase_counts=common_phase[id][phase?phase:"other"];
        for(int i=0;i<8;++i) { size_t count=after.paths[i]-before.paths[i];common_paths[id][i]+=count;phase_counts[i]+=count; }
        if (!common_m[id][batch]++) std::fprintf(stderr,"KUNLUN common dispatch PD%d: ABI=1 K=%d N=%d M=%d weight=%p GEMV=%zu GEMM=%zu exact_cl=%zu\n",
            id,k,n,batch,w,after.paths[0]-before.paths[0],after.chunks-before.chunks-(after.paths[0]-before.paths[0])-(after.paths[7]-before.paths[7]),after.paths[7]-before.paths[7]);
        return;
    }
    if (packed && packed->native) {
        bool accelerated=kunlun::native_q4_matmul(ctx->native[id],id,y,w,x,k,n,batch);
        if (accelerated) ++native_sd_total[id]; else ++native_cl_total[id];
        if (accelerated && !ctx->native_logged[id]) {
            std::fprintf(stderr,"KUNLUN native Q4: PD%d %s chunk64 CL16 K=%d N=%d batch=%d\n",id,n%256 ? "compact-n32" : "compact-n64-2d",k,n,batch);
            ctx->native_logged[id]=true;
        } else if (!accelerated && !ctx->fallback_logged[id]) {
            const char * reason=kunlun::native_q4_reason(k,n,batch,kunlun::InputType::f32);
            std::fprintf(stderr,"KUNLUN native Q4 fallback: PD%d K=%d N=%d batch=%d reason=%s (resident Q4 CL)\n",id,k,n,batch,*reason ? reason : "pointer alignment");
            ctx->fallback_logged[id]=true;
        }
        return;
    }
    auto & fn=ctx->mv[id];
    bool i8=packed && packed->packed;
    if (i8) { ++ctx->i8_mm[id]; ++native_cdnn_total[id]; }
    else if (type == 2) { ++ctx->q4_mm[id]; ++native_direct_total[id]; }
    kunlun::matmul(fn,id,y,w,x,k,n,batch,(kunlun::WeightType)(i8 ? 3 : type),
        i8 ? scales(packed,id) : nullptr,i8 ? maxima(packed,id) : nullptr,
        std::getenv("GGML_KUNLUN_BASELINE") ? 1 : 16);
}
int kernel_type(enum ggml_type t) { return t == GGML_TYPE_Q4_0 ? 2 : t == GGML_TYPE_F16 ? 1 : 0; }

void matmul(backend_context * ctx, ggml_tensor * dst) {
    const auto * w=dst->src[0]; const auto * x=dst->src[1];
    int k=w->ne[0], n=w->ne[1], batch=x->ne[1];
    auto * wb=buffer_of(w); auto * parts=(tensor_parts *)w->extra;
    const int main=ctx->id, other=1-main;
    if (!wb || !wb->type->split) {
        launch_mv(ctx,main,dst->data,w->data,x->data,k,n,batch,kernel_type(w->type),parts);
        check(kunlun::wait()); return;
    }
    GGML_ASSERT(parts && wb->type->id == main);
    const int local_rows=parts->rows[main], remote_rows=parts->rows[other];
    const int local_offset=main ? parts->rows[0] : 0;
    const int remote_offset=other ? parts->rows[0] : 0;
    if (!remote_rows) {
        launch_mv(ctx,main,dst->data,parts->ptr[main],x->data,k,n,batch,kernel_type(w->type),parts);
        check(kunlun::wait()); ++split_matmuls; return;
    }
    size_t xs=(size_t)k*batch*4, ys=(size_t)remote_rows*batch*4;
    if (ctx->split_x_size < xs) { release(other,ctx->split_x,ctx->split_x_size); ctx->split_x=allocate(other,xs); ctx->split_x_size=xs; }
    if (ctx->split_y_size < ys) { release(other,ctx->split_y,ctx->split_y_size); ctx->split_y=allocate(other,ys); ctx->split_y_size=ys; }
    GGML_ASSERT(ctx->split_x && ctx->split_y);
    void * local_output=(char *)dst->data+local_offset*4;
    if (batch > 1 && local_rows) {
        size_t size=(size_t)local_rows*batch*4;
        if (ctx->local_y_size < size) { release(main,ctx->local_y,ctx->local_y_size); ctx->local_y=allocate(main,size); ctx->local_y_size=size; }
        GGML_ASSERT(ctx->local_y); local_output=ctx->local_y;
    }
    peer(other,ctx->split_x,main,x->data,xs);
    if (local_rows) launch_mv(ctx,main,local_output,parts->ptr[main],x->data,k,local_rows,batch,kernel_type(w->type),parts);
    launch_mv(ctx,other,ctx->split_y,parts->ptr[other],ctx->split_x,k,remote_rows,batch,kernel_type(w->type),parts);
    select(other); check(kunlun::wait()); select(main); check(kunlun::wait());
    const char * copy_mode=std::getenv("GGML_KUNLUN_ROW_COPY");
    bool legacy=copy_mode && std::strcmp(copy_mode,"legacy") == 0;
    if (batch > 1 && !legacy) {
        if (local_rows) kunlun::copy_rows(ctx->gather,main,dst->data,local_output,local_rows,batch,n,local_offset);
        if (!local_rows) peer(main,dst->data,other,ctx->split_y,ys);
        else {
            if (ctx->peer_y_size < ys) { release(main,ctx->peer_y,ctx->peer_y_size); ctx->peer_y=allocate(main,ys); ctx->peer_y_size=ys; }
            GGML_ASSERT(ctx->peer_y);
            peer(main,ctx->peer_y,other,ctx->split_y,ys);
            kunlun::copy_rows(ctx->gather,main,dst->data,ctx->peer_y,remote_rows,batch,n,remote_offset);
            check(kunlun::wait());
        }
    } else {
        if (batch > 1 && local_rows) {
            for (int b=0;b<batch;++b) check(kunlun::copy((char *)dst->data+((size_t)b*n+local_offset)*4,
                (char *)local_output+(size_t)b*local_rows*4,(size_t)local_rows*4,kunlun::device_to_device));
            check(kunlun::wait());
        }
        for (int b=0;b<batch;++b) peer(main,(char *)dst->data+((size_t)b*n+remote_offset)*4,other,
            (char *)ctx->split_y+(size_t)b*remote_rows*4,(size_t)remote_rows*4);
    }
    ++split_matmuls;
}

void elementwise(backend_context * ctx, ggml_tensor * dst) {
    const void * a=dst->src[0]->data;
    const void * b=dst->src[1] ? dst->src[1]->data : a;
    int op=dst->op == GGML_OP_ADD ? 0 : dst->op == GGML_OP_MUL ? 1 : dst->op == GGML_OP_RMS_NORM ? 3 : dst->op == GGML_OP_SCALE ? 4 : 2;
    float parameter=0; std::memcpy(&parameter,dst->op_params,4);
    kunlun::elementwise(ctx->ew,ctx->id,dst->data,a,b,ggml_nelements(dst),dst->ne[0],
        dst->src[1] ? ggml_nelements(dst->src[1]) : ggml_nelements(dst),(kunlun::ElementwiseOp)op,parameter);
}

bool set_rows_layout_supported(const ggml_tensor * dst, const ggml_tensor * src, const ggml_tensor * idx) {
    size_t sb=ggml_type_size(src->type), ib=ggml_type_size(idx->type);
    if (dst->type != GGML_TYPE_F16 || (sb != 2 && sb != 4) || (ib != 4 && ib != 8) || dst->nb[0] != 2) return false;
    for (int j=0;j<4;++j) {
        if (src->ne[j] <= 0 || dst->ne[j] <= 0 || idx->ne[j] <= 0 || !src->nb[j] || !dst->nb[j] || !idx->nb[j]) return false;
        if (src->nb[j] > INT_MAX || dst->nb[j] > INT_MAX || idx->nb[j] > INT_MAX) return false;
        if (src->nb[j]%sb || dst->nb[j]%2 || idx->nb[j]%ib) return false;
    }
    size_t width=dst->ne[0]*2, rows=(dst->ne[1]-1)*dst->nb[1]+width;
    size_t heads=(dst->ne[2]-1)*dst->nb[2]+rows;
    return dst->nb[1] >= width && (dst->ne[2] == 1 || dst->nb[2] >= rows) && (dst->ne[3] == 1 || dst->nb[3] >= heads) &&
        (uintptr_t)dst->data%2 == 0 && (uintptr_t)src->data%sb == 0 && (uintptr_t)idx->data%ib == 0;
}

void set_rows_legacy(ggml_tensor * dst) {
    const auto * src = dst->src[0];
    const auto * idx = dst->src[1];
    std::vector<char> input(ggml_nbytes(src)), indices(ggml_nbytes(idx));
    ggml_backend_tensor_get(src, input.data(), 0, input.size());
    ggml_backend_tensor_get(idx, indices.data(), 0, indices.size());
    std::vector<char> scatter;
    if (src->ne[0] == 1) {
        scatter.resize(ggml_nbytes(dst));
        ggml_backend_tensor_get(dst,scatter.data(),0,scatter.size());
    }
    std::vector<float> row(src->ne[0]);
    std::vector<ggml_fp16_t> half_row(src->ne[0]);
    for (int64_t i3 = 0; i3 < src->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < src->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < src->ne[1]; ++i1) {
                size_t io = i1*idx->nb[0] + (i2%idx->ne[1])*idx->nb[1] + (i3%idx->ne[2])*idx->nb[2];
                int64_t target = 0;
                if (idx->type == GGML_TYPE_I64) std::memcpy(&target, indices.data()+io, 8);
                else { int32_t value; std::memcpy(&value, indices.data()+io, 4); target=value; }
                if (target < 0 || target >= dst->ne[1]) GGML_ABORT("SET_ROWS %s index %lld at %lld (%s, shape %lld %lld %lld %lld), destination %lld",dst->name,(long long)target,(long long)i1,ggml_type_name(idx->type),(long long)idx->ne[0],(long long)idx->ne[1],(long long)idx->ne[2],(long long)idx->ne[3],(long long)dst->ne[1]);
                size_t offset = i1*src->nb[1] + i2*src->nb[2] + i3*src->nb[3];
                for (int64_t j = 0; j < src->ne[0]; ++j) {
                    if (src->type == GGML_TYPE_F32) std::memcpy(&row[j],input.data()+offset+j*src->nb[0],4);
                    else { ggml_fp16_t h; std::memcpy(&h,input.data()+offset+j*src->nb[0],2); row[j]=ggml_fp16_to_fp32(h); }
                }
                size_t dest = target*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];
                if (dst->type == GGML_TYPE_F16) {
                    ggml_fp32_to_fp16_row(row.data(),half_row.data(),row.size());
                    if (!scatter.empty()) std::memcpy(scatter.data()+dest,half_row.data(),half_row.size()*2);
                    else ggml_backend_tensor_set(dst,half_row.data(),dest,half_row.size()*2);
                } else {
                    if (!scatter.empty()) std::memcpy(scatter.data()+dest,row.data(),row.size()*4);
                    else ggml_backend_tensor_set(dst,row.data(),dest,row.size()*4);
                }
            }
        }
    }
    if (!scatter.empty()) ggml_backend_tensor_set(dst,scatter.data(),0,scatter.size());
}

bool contiguous_cache_rows(const ggml_tensor * dst, const ggml_tensor * src, const ggml_tensor * idx,
                           const std::vector<char> & indices, kunlun::SetRowsLayout & p) {
    const auto * root=dst; while (root->view_src) root=root->view_src;
    if (dst->ne[0] != 1 || src->ne[2] != 1 || src->ne[3] != 1 || root->ne[2] != 1 || root->ne[3] != 1 ||
        root->ne[0] < 8 || root->ne[0]%8 || root->ne[1] <= 0 || root->data != dst->data ||
        !ggml_is_contiguous(root) || dst->nb[1] != 2 || src->nb[1] != ggml_type_size(src->type) ||
        idx->nb[0] != ggml_type_size(idx->type) || idx->ne[1] != 1 || idx->ne[2] != 1 ||
        dst->ne[1] != root->ne[0]*root->ne[1] || src->ne[1]%root->ne[0]) return false;
    int64_t features=root->ne[0], stride=root->ne[1], tokens=src->ne[1]/features;
    if (tokens < 1 || tokens > stride) return false;
    auto index=[&](int64_t row) {
        int64_t value;
        if (idx->type == GGML_TYPE_I64) std::memcpy(&value,indices.data()+row*idx->nb[0],8);
        else { int32_t v; std::memcpy(&v,indices.data()+row*idx->nb[0],4); value=v; }
        return value;
    };
    int64_t start=index(0);
    if (start < 0 || start+tokens > stride) return false;
    for (int64_t token=0;token<tokens;++token) for (int64_t feature=0;feature<features;++feature)
        if (index(token*features+feature) != feature*stride+start+token) return false;
    p.cache_features=features; p.cache_tokens=tokens; p.cache_stride=stride;
    return true;
}

void set_rows(backend_context * ctx, ggml_tensor * dst) {
    const auto * src=dst->src[0]; const auto * idx=dst->src[1];
    const char * mode=std::getenv("GGML_KUNLUN_SET_ROWS");
    bool supported=set_rows_layout_supported(dst,src,idx);
    uintptr_t d=(uintptr_t)dst->data, a=(uintptr_t)src->data, b=(uintptr_t)idx->data;
    bool alias=(d < a+ggml_nbytes(src) && a < d+ggml_nbytes(dst)) || (d < b+ggml_nbytes(idx) && b < d+ggml_nbytes(dst));
    supported=supported && !alias;
    if (!supported || (mode && std::strcmp(mode,"legacy") == 0)) { set_rows_legacy(dst); return; }
    std::vector<char> indices(ggml_nbytes(idx));
    ggml_backend_tensor_get(idx,indices.data(),0,indices.size());
    kunlun::SetRowsLayout p{};
    p.width=src->ne[0]; p.rows=src->ne[1]; p.dim2=src->ne[2]; p.dim3=src->ne[3];
    for (int j=0;j<4;++j) p.source_stride[j]=src->nb[j];
    for (int j=0;j<3;++j) { p.destination_stride[j]=dst->nb[j+1]; p.index_stride[j]=idx->nb[j]; }
    p.index_extent[0]=idx->ne[1]; p.index_extent[1]=idx->ne[2];
    p.source_bytes=ggml_type_size(src->type); p.index_bytes=ggml_type_size(idx->type); p.destination_rows=dst->ne[1];
    if (!contiguous_cache_rows(dst,src,idx,indices,p)) {
    std::vector<int64_t> targets(src->ne[1]);
    for (int64_t i3=0;i3<src->ne[3];++i3) for (int64_t i2=0;i2<src->ne[2];++i2) {
        for (int64_t row=0;row<src->ne[1];++row) {
            size_t off=row*idx->nb[0]+(i2%idx->ne[1])*idx->nb[1]+(i3%idx->ne[2])*idx->nb[2];
            int64_t target;
            if (idx->type == GGML_TYPE_I64) std::memcpy(&target,indices.data()+off,8);
            else { int32_t value; std::memcpy(&value,indices.data()+off,4); target=value; }
            if (target < 0 || target >= dst->ne[1]) GGML_ABORT("SET_ROWS %s index %lld outside [0,%lld)",dst->name,(long long)target,(long long)dst->ne[1]);
            targets[row]=target;
        }
        if (!std::is_sorted(targets.begin(),targets.end())) std::sort(targets.begin(),targets.end());
        if (std::adjacent_find(targets.begin(),targets.end()) != targets.end()) { set_rows_legacy(dst); return; }
    }
    }
    bool ok=kunlun::set_rows_f16(ctx->kv,ctx->id,dst->data,ggml_nbytes(dst),src->data,ggml_nbytes(src),idx->data,ggml_nbytes(idx),p);
    if (!ok) GGML_ABORT("SET_ROWS %s device index validation failed on PD%d",dst->name,ctx->id);
}

bool supports_op(ggml_backend_dev_t, const ggml_tensor * t) {
    if (t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_PERMUTE) return true;
    const auto * a = t->src[0]; const auto * b = t->src[1];
    for (const auto * src : t->src) {
        if (src) {
            auto * buf = buffer_of(src);
            if (buf && buf->type->split && (t->op != GGML_OP_MUL_MAT || (src != a && buf->size != 0))) return false;
        }
    }
    if (!a) return false;
    if (t->op == GGML_OP_SET_ROWS) {
        auto * destination = t->view_src ? t->view_src->buffer : t->buffer;
        if (destination && !is_ours(destination->buft)) return false;
        return b && (a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16) &&
            t->type == GGML_TYPE_F16 &&
            (b->type == GGML_TYPE_I64 || b->type == GGML_TYPE_I32) &&
            t->nb[0] == ggml_type_size(t->type) && t->ne[0] == a->ne[0] &&
            t->ne[1] > 0 && a->ne[0] > 0 && a->ne[0] <= 16384 &&
            t->ne[2] == a->ne[2] && t->ne[3] == a->ne[3] && b->ne[0] == a->ne[1] &&
            b->ne[1] > 0 && b->ne[2] > 0 && b->ne[3] == 1 && a->ne[2]%b->ne[1] == 0 && a->ne[3]%b->ne[2] == 0 &&
            ggml_nbytes(a) <= INT_MAX && ggml_nbytes(b) <= INT_MAX && ggml_nbytes(t) <= INT_MAX &&
            set_rows_layout_supported(t,a,b);
    }
    if (!ggml_is_contiguous(t) || !ggml_is_contiguous(a)) return false;
    if (t->op == GGML_OP_MUL_MAT) {
        return b && ggml_is_contiguous(b) && ggml_is_matrix(a) && ggml_is_matrix(b) && ggml_is_matrix(t) &&
            a->ne[0]==b->ne[0] && t->ne[0]==a->ne[1] && t->ne[1]==b->ne[1] &&
            (a->type == GGML_TYPE_Q4_0 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_F32) &&
            a->ne[0] > 0 && a->ne[0] % 32 == 0 && a->ne[0] <= 16384 && a->ne[1] > 0 && a->ne[1] <= INT_MAX && b->ne[1] > 0 &&
            b->ne[1] <= INT_MAX / std::max<int64_t>(a->ne[1], 1) && b->type == GGML_TYPE_F32 && t->type == GGML_TYPE_F32 && !a->view_src &&
            ggml_nbytes(a) <= INT_MAX && ggml_nbytes(b) <= INT_MAX && ggml_nbytes(t) <= INT_MAX;
    }
    if (a->type != GGML_TYPE_F32 || t->type != GGML_TYPE_F32 || t->ne[0] <= 0 || t->ne[0] > 16384 || ggml_nelements(t) <= 0 || ggml_nelements(t) > INT_MAX / 4) return false;
    if (t->op == GGML_OP_RMS_NORM) {
        float epsilon; std::memcpy(&epsilon,t->op_params,4);
        return t->ne[0] > 0 && t->ne[0] <= 16384 && std::isfinite(epsilon) && epsilon > 0;
    }
    if (t->op == GGML_OP_UNARY) return ggml_get_unary_op(t) == GGML_UNARY_OP_SILU;
    if (t->op == GGML_OP_SCALE) { float bias; std::memcpy(&bias, (const char *)t->op_params+4, 4); return bias == 0; }
    if (t->op == GGML_OP_ADD || t->op == GGML_OP_MUL) {
        return b && b->type == GGML_TYPE_F32 && ggml_is_contiguous(b) &&
            (ggml_are_same_shape(a, b) || (ggml_nelements(b) == a->ne[0] && ggml_is_vector(b)));
    }
    return false;
}

ggml_backend_t init_backend(ggml_backend_dev_t dev, const char *) {
    auto * ctx = new backend_context{}; ctx->id = ((device_context *)dev->context)->id;
    static ggml_guid guid = {0xc3,0xe7,0x03,0x2b,0x9d,0xd1,0x4c,0xc2,0xac,0x8b,0x12,0x5b,0x78,0x32,0x2f,0x90};
    ggml_backend_i iface{};
    iface.get_name = [](ggml_backend_t b) { return ggml_backend_dev_name(b->device); };
    iface.free = [](ggml_backend_t b) {
        std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
        auto * c = (backend_context *)b->context;
        release(1-c->id,c->split_x,c->split_x_size); release(1-c->id,c->split_y,c->split_y_size);
        release(c->id,c->local_y,c->local_y_size);
        release(c->id,c->peer_y,c->peer_y_size);
        auto stats=kunlun::statistics();
        GGML_LOG_INFO("KUNLUN stats: pd0_mm=%zu pd1_mm=%zu split_mm=%zu p2p_copies=%zu p2p_bytes=%zu\n",stats.matmuls[0],stats.matmuls[1],split_matmuls.load(),stats.peer_copies,stats.peer_bytes);
        for (int id=0;id<2;++id) {
            auto common=c->common[id].statistics();
            std::fprintf(stderr,"KUNLUN common workspace PD%d: calls=%zu chunks=%zu hbm=%zu l3=%zu slice=%zu tracked_hbm=%zu\n",id,common.calls,common.chunks,common.hbm_workspace_bytes,common.l3_workspace_bytes,common.slice_bytes,kunlun::allocated_bytes(id));
            auto s=c->native[id].statistics();
            GGML_LOG_INFO("KUNLUN paths PD%d: native_sd=%zu native_cl=%zu cdnn=%zu direct_q4=%zu scratch_hbm=%zu scratch_l3=%zu native_uploads=%zu native_upload_bytes=%zu native_setup_ms=%.3f hbm_tracked=%zu\n",
                id,s.accelerated,s.fallback,c->i8_mm[id],c->q4_mm[id],s.hbm_workspace_bytes,s.l3_workspace_bytes,
                native_uploads[id].load(),native_upload_bytes[id].load(),native_upload_ns[id].load()/1e6,kunlun::allocated_bytes(id));
        }
        delete c; delete b;
    };
    iface.graph_compute = [](ggml_backend_t b, ggml_cgraph * graph) {
        std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
        auto * c = (backend_context *)b->context;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * t = ggml_graph_node(graph,i);
            kunlun::diagnostic_tensor(t->name,t->ne);
            if (t->op == GGML_OP_MUL_MAT) matmul(c,t);
            else if (t->op == GGML_OP_SET_ROWS) set_rows(c,t);
            else if (t->op == GGML_OP_ADD || t->op == GGML_OP_MUL || t->op == GGML_OP_RMS_NORM || t->op == GGML_OP_UNARY || t->op == GGML_OP_SCALE) elementwise(c,t);
            else GGML_ASSERT(t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE);
        }
        kunlun::diagnostic_tensor(nullptr);
        return GGML_STATUS_SUCCESS;
    };
    return new ggml_backend{&guid,iface,dev,ctx};
}

ggml_backend_buffer_type_t split_type(int main_device, const float * fractions) {
    if (main_device < 0 || main_device > 1) return nullptr;
    const char * mode=std::getenv("GGML_KUNLUN_ROW_SPLIT");
    if (main_device == 1 && mode && std::strcmp(mode,"legacy") == 0) return nullptr;
    int count = 0; if (kunlun::device_count(&count) || count != 2) return nullptr;
    double f = .5;
    if (fractions) {
        if (!std::isfinite(fractions[0]) || !std::isfinite(fractions[1]) || fractions[0] < 0 || fractions[1] < 0) return nullptr;
        double total = (double)fractions[0] + fractions[1];
        if (total > 0) f = fractions[0] / total;
    }
    static std::vector<std::unique_ptr<type_context>> contexts;
    static std::vector<std::unique_ptr<ggml_backend_buffer_type>> types;
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
    for (size_t i=0;i<contexts.size();++i) if (contexts[i]->id == main_device && contexts[i]->fraction == f) return types[i].get();
    contexts.emplace_back(new type_context{main_device,true,f,"KUNLUN_Split"+std::to_string(main_device)});
    types.emplace_back(new ggml_backend_buffer_type{type_iface,&devices[main_device],contexts.back().get()});
    return types.back().get();
}
} // namespace

ggml_backend_reg_t ggml_backend_kunlun_reg(void) {
    static device_context dc[2] = {{0,"KUNLUN0"},{1,"KUNLUN1"}};
    static ggml_backend_reg reg{};
    static std::once_flag once;
    std::call_once(once, [] {
        std::atexit(log_native_totals);
        std::atexit(log_common_totals);
        const char * common_setting=std::getenv("GGML_KUNLUN_COMMON_Q4");
        if (common_setting && std::strcmp(common_setting,"0") && std::strcmp(common_setting,"1"))
            std::fprintf(stderr,"KUNLUN common Q4 disabled: invalid GGML_KUNLUN_COMMON_Q4=%s (expected 0 or 1)\n",common_setting);
        if (common_enabled()) std::fprintf(stderr,"KUNLUN common Q4 opt-in: common ABI v1 > legacy native > requant I8; M>1 Row A8, M=1 two-component A8\n");
        type_iface.get_name = [](ggml_backend_buffer_type_t b) { return ((type_context *)b->context)->name.c_str(); };
        type_iface.alloc_buffer = alloc_buffer;
        type_iface.get_alignment = [](ggml_backend_buffer_type_t) -> size_t { return 256; };
        // KL1 HBM has separate address regions; avoid a single multi-GB allocation.
        type_iface.get_max_size = [](ggml_backend_buffer_type_t) -> size_t { return 1ULL << 30; };
        type_iface.get_alloc_size = [](ggml_backend_buffer_type_t b, const ggml_tensor * t) {
            if (((type_context *)b->context)->split) return ggml_nbytes(t);
            if (use_common(t)) return kunlun::common_q4_size(t->ne[0],t->ne[1]);
            if (use_native(t)) return kunlun::native_q4_size(t->ne[0],t->ne[1]);
            return use_i8(t) ? packed_size(t->ne[0],t->ne[1]) : ggml_nbytes(t);
        };
        buffer_iface.get_base = [](ggml_backend_buffer_t b) { return ((buffer_context *)b->context)->base; };
        buffer_iface.init_tensor = [](ggml_backend_buffer_t b, ggml_tensor * t) {
            auto * c = (buffer_context *)b->context;
            if (t->view_src) return GGML_STATUS_SUCCESS;
            const char * native_setting=std::getenv("GGML_KUNLUN_NATIVE_Q4");
            if (native_setting && std::strcmp(native_setting,"1")==0 && t->type==GGML_TYPE_Q4_0 && !use_native(t) && !use_common(t))
                std::fprintf(stderr,"KUNLUN native Q4 storage fallback: tensor=%s K=%lld N=%lld reason=outside validated native shape/layout method=%s\n",
                    t->name,(long long)t->ne[0],(long long)t->ne[1],use_i8(t)?"CDNN":"existing Q4");
            if (common_enabled() && t->type==GGML_TYPE_Q4_0 && !use_common(t))
                std::fprintf(stderr,"KUNLUN common storage fallback: tensor=%s K=%lld N=%lld reason=type/shape/stride/view method=%s\n",t->name,(long long)t->ne[0],(long long)t->ne[1],use_native(t)?"legacy native":use_i8(t)?"CDNN":"original Q4");
            if (!c->type->split && !use_i8(t) && !use_native(t) && !use_common(t)) return GGML_STATUS_SUCCESS;
            std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
            auto p = std::unique_ptr<tensor_parts>(new tensor_parts{});
            GGML_ASSERT(ggml_is_matrix(t));
            p->packed=use_i8(t); p->native=use_native(t);p->common=use_common(t); p->k=t->ne[0];
            if (p->packed || p->native || p->common) p->original.resize(ggml_nbytes(t));
            if (c->type->split) { p->rows[0] = t->ne[1] * c->type->fraction; p->rows[1] = t->ne[1] - p->rows[0]; }
            else p->rows[c->type->id]=t->ne[1];
            for (int d=0;d<2;++d) {
                p->bytes[d] = (size_t)p->rows[d]*ggml_row_size(t->type,t->ne[0]);
                p->physical[d] = p->common ? kunlun::common_q4_size(p->k,p->rows[d]) : p->native ? kunlun::native_q4_size(p->k,p->rows[d]) : p->packed ? packed_size(p->k,p->rows[d]) : p->bytes[d];
                if (p->bytes[d]) {
                    p->ptr[d] = c->type->split ? allocate(d,p->physical[d]) : t->data;
                    if (!p->ptr[d]) { for(int j=0;j<d;++j) { p->common_weight[j].reset();release(j,p->ptr[j],p->physical[j]); } return GGML_STATUS_ALLOC_FAILED; }
                    if (p->common) {
                        p->common_weight[d].reset(new kunlun::CommonQ4Weight(d,p->k,p->rows[d],p->ptr[d],p->physical[d]));
                        ++common_regions[d];common_region_bytes[d]+=p->physical[d];
                    }
                }
            }
            t->extra = p.get(); c->tensors.push_back(std::move(p)); return GGML_STATUS_SUCCESS;
        };
        buffer_iface.free_buffer = [](ggml_backend_buffer_t b) {
            std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
            auto * c = (buffer_context *)b->context;
            if (c->type->split) {
                for (auto & p : c->tensors) for (int d=0;d<2;++d) { p->common_weight[d].reset();release(d,p->ptr[d],p->physical[d]); }
                munmap(c->base,std::max<size_t>(c->size,1));
            } else { c->tensors.clear();release(c->type->id,c->base,c->size); }
            delete c;
        };
        buffer_iface.set_tensor = [](ggml_backend_buffer_t b, ggml_tensor * t, const void * p, size_t o, size_t s) { transfer(b,t,const_cast<void *>(p),o,s,true); };
        buffer_iface.get_tensor = [](ggml_backend_buffer_t b, const ggml_tensor * t, void * p, size_t o, size_t s) { transfer(b,t,p,o,s,false); };
        buffer_iface.memset_tensor = [](ggml_backend_buffer_t b, ggml_tensor * t, uint8_t v, size_t o, size_t s) { std::vector<char> data(s,v); transfer(b,t,data.data(),o,s,true); };
        buffer_iface.clear = [](ggml_backend_buffer_t b, uint8_t v) {
            std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
            auto * c = (buffer_context *)b->context;
            if (c->type->split) {
                for (auto & p : c->tensors) {
                    if (p->packed || p->native || p->common) { std::fill(p->original.begin(),p->original.end(),v); repack(p.get(),0,p->rows[0]+p->rows[1]); }
                    else for (int d=0;d<2;++d) clear_device(d,p->ptr[d],p->physical[d],v);
                }
                return;
            }
            clear_device(c->type->id,c->base,c->size,v);
            for (auto & p : c->tensors) if (p->packed || p->native || p->common) { std::fill(p->original.begin(),p->original.end(),v); repack(p.get(),0,p->rows[0]+p->rows[1]); }
        };
        buffer_iface.cpy_tensor = [](ggml_backend_buffer_t b, const ggml_tensor * src, ggml_tensor * dst) {
            auto * s = buffer_of(src); auto * d = (buffer_context *)b->context;
            if (!s || s->type->split || d->type->split) return false;
            const auto * sr = src; const auto * dr = dst;
            while (sr->view_src) sr=sr->view_src;
            while (dr->view_src) dr=dr->view_src;
            if ((sr->type == GGML_TYPE_Q4_0 && sr->extra) || (dr->type == GGML_TYPE_Q4_0 && dr->extra)) return false;
            std::lock_guard<std::recursive_mutex> lock(runtime_mutex);
            if(s->type->id == d->type->id) { select(d->type->id); check(kunlun::copy(dst->data,src->data,ggml_nbytes(src),kunlun::device_to_device)); check(kunlun::wait()); }
            else peer(d->type->id,dst->data,s->type->id,src->data,ggml_nbytes(src));
            return true;
        };
        ggml_backend_device_i di{};
        di.get_name = [](ggml_backend_dev_t d) { return ((device_context *)d->context)->name.c_str(); };
        di.get_description = [](ggml_backend_dev_t) { return "Kunlun K200 KL1"; };
        di.get_memory = [](ggml_backend_dev_t d, size_t * f, size_t * t) { *t = 8064ULL<<20; size_t used=kunlun::allocated_bytes(((device_context *)d->context)->id); *f=used < *t ? *t-used : 0; };
        di.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        di.get_props = [](ggml_backend_dev_t d, ggml_backend_dev_props * p) { *p={}; p->name=d->iface.get_name(d); p->description=d->iface.get_description(d); d->iface.get_memory(d,&p->memory_free,&p->memory_total); p->type=GGML_BACKEND_DEVICE_TYPE_GPU; };
        di.init_backend = init_backend;
        di.get_buffer_type = [](ggml_backend_dev_t d) { return &normal_types[((device_context *)d->context)->id]; };
        di.supports_op = supports_op;
        di.supports_buft = [](ggml_backend_dev_t d, ggml_backend_buffer_type_t b) { return is_ours(b) && ((type_context *)b->context)->id == ((device_context *)d->context)->id; };
        di.offload_op = [](ggml_backend_dev_t d, const ggml_tensor * t) { return supports_op(d,t) && t->op == GGML_OP_MUL_MAT; };
        reg.api_version=GGML_BACKEND_API_VERSION;
        reg.iface.get_name=[](ggml_backend_reg_t) { return "KUNLUN"; };
        reg.iface.get_device_count=[](ggml_backend_reg_t) -> size_t { int n=0; return kunlun::device_count(&n)==0 ? std::min(n,2) : 0; };
        reg.iface.get_device=[](ggml_backend_reg_t, size_t i) { GGML_ASSERT(i<2); return &devices[i]; };
        reg.iface.get_proc_address=[](ggml_backend_reg_t, const char * name) -> void * { return std::strcmp(name,"ggml_backend_split_buffer_type")==0 ? (void *)split_type : nullptr; };
        for(int i=0;i<2;++i) {
            devices[i]={di,&reg,&dc[i]}; normal_contexts[i]={i,false,1.0,dc[i].name};
            normal_types[i]={type_iface,&devices[i],&normal_contexts[i]};
        }
    });
    return &reg;
}
GGML_BACKEND_DL_IMPL(ggml_backend_kunlun_reg)

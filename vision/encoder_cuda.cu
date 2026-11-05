/* The Qwen3.8 vision tower on CUDA, extracted from DwarfStar's
 * ds4_qwen4_cuda.cuh (vision kernels, dense projections) and ds4_cuda.cu
 * (runtime glue) (MIT). */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "encoder.h"

namespace {

struct gpu_tensor {
    void *ptr;
    uint64_t bytes;
};

enum { VIS_BUFFERS = 14 };
const uint64_t WEIGHT_COPY_CHUNK = 64ull * 1024ull * 1024ull;

struct gpu_state {
    bool open;
    cublasHandle_t cublas;
    char *weights;  /* the file's bytes [data_pos, size) */
    uint64_t data_pos, size;
    void *tmp;
    uint64_t tmp_bytes;
    gpu_tensor buf[VIS_BUFFERS];
    uint32_t buf_patches;
    char error[256];
};

gpu_state g;

void gpu_error(const char *fmt, const char *what, const char *detail) {
    if (!g.error[0]) snprintf(g.error, sizeof(g.error), fmt, what, detail);
}

int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    gpu_error("CUDA %s failed: %s", what, cudaGetErrorString(err));
    return 0;
}

int cublas_ok(cublasStatus_t st, const char *what) {
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    gpu_error("cuBLAS %s failed: %s", what, cublasGetStatusString(st));
    return 0;
}

/* ds4 runs the encoder on the legacy default stream and its default cuBLAS handle. */
cudaStream_t cuda_decode_stream(void) { return (cudaStream_t)0; }
cublasHandle_t cuda_cublas_for_tier(int) { return g.cublas; }

/* One scratch slab, grown on demand and kept. */
void *cuda_tmp_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g.tmp_bytes >= bytes) return g.tmp;
    if (g.tmp) {
        if (!cuda_ok(cudaDeviceSynchronize(), "synchronize CUDA scratch growth")) return NULL;
        (void)cudaFree(g.tmp);
        g.tmp = NULL;
        g.tmp_bytes = 0;
    }
    void *ptr = NULL;
    if (!cuda_ok(cudaMalloc(&ptr, (size_t)bytes), what)) {
        (void)cudaGetLastError();
        return NULL;
    }
    g.tmp = ptr;
    g.tmp_bytes = bytes;
    return g.tmp;
}

int gpu_tensor_write(gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy((char *)tensor->ptr + offset, data, (size_t)bytes, cudaMemcpyHostToDevice),
                   "tensor write");
}

int gpu_tensor_read(const gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy(data, (const char *)tensor->ptr + offset, (size_t)bytes, cudaMemcpyDeviceToHost),
                   "tensor read");
}

/* ds4's DS4_CUDA_END_STREAM_SYNC variant: it does not wait for other streams in the process. */
int gpu_end_commands(void) { return cuda_ok(cudaStreamSynchronize(0), "end commands stream"); }

} // namespace

namespace qwen4_cuda {

static bool tensor(const gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && bytes <= t->bytes;
}

static uint64_t row_bytes(uint32_t type, uint64_t n) {
    switch (type) {
    case 0: return n * 4;
    case 1: case 30: return n * 2;
    case 8: return n % 32 ? 0 : n / 32 * 34;
    default: return 0;
    }
}

/* off is a file offset; every tensor is resident from open. */
static const char *weight(uint64_t off, uint64_t bytes) {
    if (!g.weights || !bytes || off < g.data_pos || off > g.size || bytes > g.size - off) return NULL;
    return g.weights + (off - g.data_pos);
}

static int launched(void) { return cuda_ok(cudaGetLastError(), "Qwen kernel"); }

__device__ __forceinline__ float sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
    return x;
}

__device__ __forceinline__ float block_sum(float x, float *shared) {
    x = sum(x);
    if (!(threadIdx.x & 31)) shared[threadIdx.x / 32] = x;
    __syncthreads();
    float result = 0;
    for (unsigned i = 0; i < blockDim.x / 32; i++) result += shared[i];
    __syncthreads();
    return result;
}

/* Templates remove unused formats from each kernel. */
template<unsigned TYPE>
__device__ __forceinline__ float value(const char *row, unsigned i) {
    if (TYPE == 0) return ((const float *)row)[i];
    if (TYPE == 1) return __half2float(((const __half *)row)[i]);
    if (TYPE == 30) return __bfloat162float(((const __nv_bfloat16 *)row)[i]);
    if (TYPE == 8) {
        const char *b = row + (i / 32) * 34;
        return __half2float(*(const __half *)b) * (float)((const int8_t *)b)[2 + i % 32];
    }
    return 0;
}

template<unsigned TYPE>
__global__ void unpack(float *out, const char *w, unsigned K, unsigned M, uint64_t rb) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < (uint64_t)K*M) out[i] = value<TYPE>(w+(i/K)*rb,i%K);
}

/* Power-of-two row scaling protects half range. Preserve the residual too,
 * so dense projections retain substantially more than half precision. */
template<unsigned TYPE>
__global__ void pack_half_components(float *scales, __half *hi, __half *lo,
        const char *x, unsigned K, uint64_t rb) {
    const unsigned row = blockIdx.x, tid = threadIdx.x;
    __shared__ float maxima[256], inv;
    float mx = 0;
    for (unsigned k = tid; k < K; k += 256) mx = fmaxf(mx,fabsf(value<TYPE>(x+(uint64_t)row*rb,k)));
    maxima[tid] = mx;
    __syncthreads();
    for (unsigned d = 128; d; d /= 2) {
        if (tid < d) maxima[tid] = fmaxf(maxima[tid],maxima[tid+d]);
        __syncthreads();
    }
    if (!tid) {
        const int e = maxima[0] > 0 ? max(-120,min(120,(int)((__float_as_uint(maxima[0])>>23)&255)-127)) : 0;
        inv = ldexpf(1,-e); scales[row] = ldexpf(1,e);
    }
    __syncthreads();
    for (unsigned k = tid; k < K; k += 256) {
        const float v = value<TYPE>(x+(uint64_t)row*rb,k)*inv;
        const __half h = __float2half_rn(v);
        hi[(uint64_t)row*K+k] = h;
        lo[(uint64_t)row*K+k] = __float2half_rn((v-__half2float(h))*4096);
    }
}

__global__ void dense_rescale(float *out, const float *scales,
        unsigned M, unsigned N) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i < (uint64_t)M*N) out[i] *= scales[i/M];
}

static int dense_f16_blas(float *out, const float *x, const __half *w,
        unsigned T, unsigned K, unsigned M) {
    const uint64_t xn = (uint64_t)T*K;
    __half *hi = (__half *)cuda_tmp_alloc((xn+T)*4, "Qwen F16 projection scratch");
    if (!hi) return 0;
    __half *lo = hi+xn;
    float *scales = (float *)(lo+xn);
    pack_half_components<0><<<T,256,0,cuda_decode_stream()>>>(scales,hi,lo,(const char *)x,K,(uint64_t)K*4);
    if (!launched()) return 0;
    const float zero = 0, one = 1, low = 0x1p-12f;
    if (!cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                M,T,K,&low,w,CUDA_R_16F,K,lo,CUDA_R_16F,K,&zero,out,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT), "Qwen F16 residual projection") ||
        !cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                M,T,K,&one,w,CUDA_R_16F,K,hi,CUDA_R_16F,K,&one,out,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT), "Qwen F16 leading projection")) return 0;
    dense_rescale<<<((uint64_t)M*T+255)/256,256,0,cuda_decode_stream()>>>(out,scales,M,T);
    return launched();
}

__global__ void dense_rescale2(float *out, const float *xs, const float *ws,
        unsigned M, unsigned T, unsigned stride) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i < (uint64_t)M*T) {
        const uint64_t dst = (i/M)*stride+i%M;
        out[dst] = out[dst]*xs[i/M]*ws[i%M];
    }
}

static int dense_q8_blas(float *out, const float *x, const char *w,
        unsigned T, unsigned K, unsigned M) {
    const unsigned bound = (unsigned)std::max(UINT64_C(1),(UINT64_C(32)<<20)/((uint64_t)K*4));
    const unsigned tile = std::min(M,bound >= 64 ? bound/64*64 : bound);
    const uint64_t xn = (uint64_t)T*K, wn = (uint64_t)tile*K;
    __half *xh = (__half *)cuda_tmp_alloc((xn+wn+T+tile)*4,"Qwen Q8 projection scratch");
    if (!xh) return 0;
    __half *xl = xh+xn, *wh = xl+xn, *wl = wh+wn;
    float *xs = (float *)(wl+wn), *ws = xs+T;
    pack_half_components<0><<<T,256,0,cuda_decode_stream()>>>(xs,xh,xl,(const char *)x,K,(uint64_t)K*4);
    if (!launched()) return 0;
    const float zero = 0, one = 1, low = 0x1p-12f;
    const uint64_t rb = row_bytes(8,K);
    for (unsigned r = 0; r < M; r += tile) {
        const unsigned n = std::min(tile,M-r);
        pack_half_components<8><<<n,256,0,cuda_decode_stream()>>>(ws,wh,wl,w+(uint64_t)r*rb,K,rb);
        if (!launched()) return 0;
        if (!cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&low,wl,CUDA_R_16F,K,xh,CUDA_R_16F,K,&zero,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT),"Qwen Q8 weight residual") ||
            !cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&low,wh,CUDA_R_16F,K,xl,CUDA_R_16F,K,&one,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT),"Qwen Q8 input residual") ||
            !cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&one,wh,CUDA_R_16F,K,xh,CUDA_R_16F,K,&one,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT),"Qwen Q8 leading product")) return 0;
        dense_rescale2<<<((uint64_t)n*T+255)/256,256,0,cuda_decode_stream()>>>(out+r,xs,ws,n,T,M);
        if (!launched()) return 0;
    }
    return 1;
}

/* Bound the temporary weight expansion instead of retaining an FP32 copy
 * of each projection. cuBLAS uses FP32 math, including FP32 activations. */
static int dense_blas(float *out, const float *x, const char *w,
        unsigned type, unsigned T, unsigned K, unsigned M) {
    const unsigned tile = std::min(M, (unsigned)std::max(UINT64_C(1), (UINT64_C(64)*1024*1024)/((uint64_t)K*4)));
    float *scratch = type ? (float *)cuda_tmp_alloc((uint64_t)tile*K*4, "Qwen dense FP32 tile") : NULL;
    if (type && !scratch) return 0;
    const uint64_t rb = row_bytes(type,K);
    const float alpha = 1, beta = 0;
    for (unsigned r = 0; r < M; r += tile) {
        const unsigned n = std::min(tile,M-r);
        const float *wf = type ? scratch : (const float *)w+(uint64_t)r*K;
        if (type) {
#define QWEN_UNPACK(TYPE) case TYPE: unpack<TYPE><<<((uint64_t)n*K+255)/256,256,0,cuda_decode_stream()>>>(scratch,w+(uint64_t)r*rb,K,n,rb); break
            switch (type) {
            QWEN_UNPACK(1); QWEN_UNPACK(8); QWEN_UNPACK(30);
            default: return 0;
            }
#undef QWEN_UNPACK
            if (!launched()) return 0;
        }
        if (!cublas_ok(cublasGemmEx(cuda_cublas_for_tier(0),CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&alpha,wf,CUDA_R_32F,K,x,CUDA_R_32F,K,&beta,out+r,CUDA_R_32F,M,
                CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT), "Qwen FP32 projection")) return 0;
    }
    return 1;
}

__global__ void vis_patch(float *x, const float *a, const float *b, const float *bias,
        const float *pos, unsigned N, unsigned E) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (uint64_t)N * E) x[i] = a[i] + b[i] + bias[i % E] + pos[i];
}

__global__ void vis_norm(float *out, const float *x, const float *w, const float *b, unsigned E, float eps) {
    const unsigned tid = threadIdx.x;
    const uint64_t base = (uint64_t)blockIdx.x * E;
    __shared__ float red[32];
    float v = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) v += x[base + i];
    const float mean = block_sum(v, red) / E;
    v = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) { const float d = x[base + i] - mean; v += d*d; }
    const float inv = rsqrtf(block_sum(v, red) / E + eps);
    for (unsigned i = tid; i < E; i += blockDim.x) out[base + i] = (x[base + i] - mean) * inv * w[i] + b[i];
}

__global__ void vis_qkv(float *q, float *k, float *v, const float *qkv, const float *bias,
        unsigned H, unsigned D, unsigned grid_w) {
    const unsigned row = blockIdx.x, tid = threadIdx.x, E = H*D, hd = D/2, qd = D/4;
    const unsigned blk = row/4, within = row%4, w2 = grid_w/2;
    const float hp = (blk/w2)*2 + within/2, wp = (blk%w2)*2 + within%2;
    const float *src = qkv + (uint64_t)row*3*E;
    const uint64_t off = (uint64_t)row*E;
    for (unsigned idx = tid; idx < 2*H*hd; idx += blockDim.x) {
        const unsigned part = idx/(H*hd), rem = idx%(H*hd), h = rem/hd, i = rem%hd;
        const unsigned base = part*E + h*D;
        const float x0 = src[base+i] + bias[base+i], x1 = src[base+i+hd] + bias[base+i+hd];
        const unsigned j = i < qd ? i : i-qd;
        const float theta = (i < qd ? hp : wp) * powf(10000.0f, -2.0f*j/hd);
        float s, c; sincosf(theta, &s, &c);
        float *dst = part ? k : q;
        dst[off+h*D+i] = x0*c - x1*s;
        dst[off+h*D+i+hd] = x0*s + x1*c;
    }
    for (unsigned i = tid; i < E; i += blockDim.x) v[off+i] = src[2*E+i] + bias[2*E+i];
}

__global__ void vis_attention(float *out, const float *q, const float *k, const float *v,
        unsigned N, unsigned H, unsigned D) {
    const unsigned row = blockIdx.x, h = blockIdx.y, lane = threadIdx.x, E = H*D;
    const uint64_t qb = (uint64_t)row*E + h*D;
    float query[3] = {}, acc[3] = {};
    for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) query[j] = q[qb+lane+j*32];
    float mx = -INFINITY, denom = 0;
    for (unsigned p = 0; p < N; p++) {
        const uint64_t kb = (uint64_t)p*E + h*D;
        float score = 0;
        for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) score += query[j]*k[kb+lane+j*32];
        score = sum(score) * rsqrtf((float)D);
        const float nm = fmaxf(mx, score), old = expf(mx-nm), prob = expf(score-nm);
        denom = denom*old + prob;
        for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) acc[j] = acc[j]*old + prob*v[kb+lane+j*32];
        mx = nm;
    }
    for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) out[qb+lane+j*32] = acc[j]/denom;
}

__global__ void vis_add(float *x, const float *add, const float *bias, unsigned N, unsigned E, unsigned mode) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (uint64_t)N*E) return;
    if (add) { x[i] += add[i] + bias[i%E]; return; }
    float t = x[i] + bias[i%E];
    if (!mode) t *= .5f * (1 + tanhf(fminf(30, fmaxf(-30, .7978845608f*(t+.044715f*t*t*t)))));
    else if (mode == 1) t *= .5f * (1 + erff(t*.70710678f));
    x[i] = t;
}

} // namespace qwen4_cuda

/* ds4's matvec (T <= 8) and cuBLAS-less tile paths are not carried: every
 * vision projection has T >= 64 rows, and open fails without cuBLAS. */
static int dense_mm_tensor(gpu_tensor *out, const gpu_tensor *x, uint64_t off, uint32_t type,
        uint32_t T, uint32_t K, uint32_t M) {
    using namespace qwen4_cuda;
    if (!T || !K || !M || !tensor(x, (uint64_t)T*K*4) || !tensor(out, (uint64_t)T*M*4)) return 0;
    const uint64_t rb = row_bytes(type,K);
    if (!rb) return 0;
    const char *w = weight(off, rb*M);
    if (!w) return 0;
    if (T < 32 || T > INT_MAX || K > INT_MAX || M > INT_MAX) return 0;
    if (type == 1)
        return dense_f16_blas((float *)out->ptr,(const float *)x->ptr,(const __half *)w,T,K,M);
    if (type == 8)
        return dense_q8_blas((float *)out->ptr,(const float *)x->ptr,w,T,K,M);
    return dense_blas((float *)out->ptr,(const float *)x->ptr,w,type,T,K,M);
}

/* ds4 allocates these per encode; here they are kept, sized for the largest
 * image so far (from open: the largest the token limit allows). */
static void buffer_floats(uint32_t N, const renza_qwen4_vision_weights *w, uint64_t n[VIS_BUFFERS]) {
    const uint64_t E = w->n_embd, FF = w->n_ff, P = w->n_patch, M = w->n_merge, O = w->n_out;
    const uint64_t IP = 3*P*P, ME = E*M*M, merged = N/(M*M);
    const uint64_t sizes[VIS_BUFFERS] = {
        N*IP, N*E, N*E, N*E, N*E, N*E, N*3*E, N*E, N*E, N*E, N*E, N*FF, merged*ME, merged*O,
    };
    memcpy(n, sizes, sizeof(sizes));
}

static void free_buffers(void) {
    for (unsigned i = 0; i < VIS_BUFFERS; i++) {
        if (g.buf[i].ptr) (void)cudaFree(g.buf[i].ptr);
        g.buf[i].ptr = NULL;
        g.buf[i].bytes = 0;
    }
    g.buf_patches = 0;
}

static int ensure_buffers(uint32_t N, const renza_qwen4_vision_weights *w) {
    if (N <= g.buf_patches) return 1;
    free_buffers();
    uint64_t n[VIS_BUFFERS];
    buffer_floats(N, w, n);
    for (unsigned i = 0; i < VIS_BUFFERS; i++) {
        if (!cuda_ok(cudaMalloc(&g.buf[i].ptr, (size_t)(n[i]*4)), "vision buffer alloc")) {
            g.buf[i].ptr = NULL;
            free_buffers();
            return 0;
        }
        g.buf[i].bytes = n[i]*4;
    }
    g.buf_patches = N;
    return 1;
}

static int vision_encode(float *out, const float *patches, const float *pos_embed,
        uint32_t N, uint32_t grid_w, const renza_qwen4_vision_weights *w) {
    using namespace qwen4_cuda;
    if (!g.open || !out || !patches || !pos_embed || !w || !N || N > 65535) return 0;
    const unsigned E = w->n_embd, FF = w->n_ff, H = w->n_head, D = H ? E/H : 0;
    const unsigned P = w->n_patch, M = w->n_merge, O = w->n_out;
    if (M != 2 || !grid_w || grid_w%M || N%grid_w || (N/grid_w)%M ||
        (D != 64 && D != 72) || H*D != E || !FF || !O || !P || P > 32 || E%32) return 0;
    const unsigned IP = 3*P*P, ME = E*M*M, merged = N/(M*M);
    if (!ensure_buffers(N, w)) return 0;
    gpu_tensor *patch = &g.buf[0], *a0 = &g.buf[1], *a1 = &g.buf[2];
    gpu_tensor *pos = &g.buf[3], *x = &g.buf[4], *tmp = &g.buf[5];
    gpu_tensor *qkv = &g.buf[6], *q = &g.buf[7], *k = &g.buf[8];
    gpu_tensor *v = &g.buf[9], *attn = &g.buf[10], *ffn = &g.buf[11];
    gpu_tensor *m0 = &g.buf[12], *res = &g.buf[13];
    bool ok = true;
    auto ptr = [](gpu_tensor *t) { return (float *)t->ptr; };
    auto wf = [&](uint64_t off, unsigned n) { return (const float *)weight(off,(uint64_t)n*4); };
    auto mm = [&](gpu_tensor *dst, const gpu_tensor *src, uint64_t off, unsigned type,
                  unsigned rows, unsigned K, unsigned M) {
        return dense_mm_tensor(dst,src,off,type,rows,K,M);
    };
    auto norm = [&](gpu_tensor *dst, const gpu_tensor *src, uint64_t wo, uint64_t bo) {
        const float *wgt = wf(wo,E), *bias = wf(bo,E);
        if (!wgt || !bias) return 0;
        vis_norm<<<N,256,0,cuda_decode_stream()>>>(ptr(dst),(const float *)src->ptr,wgt,bias,E,w->eps);
        return launched();
    };
    auto add = [&](gpu_tensor *dst, const gpu_tensor *src, uint64_t bo, unsigned rows,
                   unsigned width, unsigned mode) {
        const float *bias = wf(bo,width);
        if (!bias) return 0;
        vis_add<<<((uint64_t)rows*width+255)/256,256,0,cuda_decode_stream()>>>(ptr(dst),
            src ? (const float *)src->ptr : NULL,bias,rows,width,mode);
        return launched();
    };
    do {
        ok = gpu_tensor_write(patch,0,patches,(uint64_t)N*IP*4) &&
             gpu_tensor_write(pos,0,pos_embed,(uint64_t)N*E*4);
        if (!ok) break;
        ok = mm(a0,patch,w->patch_w0,w->patch_type,N,IP,E) && mm(a1,patch,w->patch_w1,w->patch_type,N,IP,E);
        const float *pb = wf(w->patch_b,E);
        if (!ok || !pb) { ok = false; break; }
        vis_patch<<<((uint64_t)N*E+255)/256,256,0,cuda_decode_stream()>>>(ptr(x),ptr(a0),ptr(a1),pb,ptr(pos),N,E);
        ok = launched();
        for (unsigned l = 0; ok && l < RENZA_QWEN4_VISION_LAYERS; l++) {
            const auto &lw = w->layer[l];
            ok = norm(tmp,x,lw.ln1_w,lw.ln1_b) && mm(qkv,tmp,lw.qkv_w,lw.qkv_type,N,E,3*E);
            const float *qb = wf(lw.qkv_b,3*E);
            if (!ok || !qb) { ok = false; break; }
            vis_qkv<<<N,256,0,cuda_decode_stream()>>>(ptr(q),ptr(k),ptr(v),ptr(qkv),qb,H,D,grid_w);
            vis_attention<<<dim3(N,H),32,0,cuda_decode_stream()>>>(ptr(attn),ptr(q),ptr(k),ptr(v),N,H,D);
            ok = launched() && mm(tmp,attn,lw.out_w,lw.out_type,N,E,E) && add(x,tmp,lw.out_b,N,E,2) &&
                 norm(tmp,x,lw.ln2_w,lw.ln2_b) && mm(ffn,tmp,lw.up_w,lw.up_type,N,E,FF) &&
                 add(ffn,NULL,lw.up_b,N,FF,0) && mm(tmp,ffn,lw.down_w,lw.down_type,N,FF,E) &&
                 add(x,tmp,lw.down_b,N,E,2);
        }
        ok = ok && norm(tmp,x,w->post_ln_w,w->post_ln_b) && mm(m0,tmp,w->mm0_w,w->mm0_type,merged,ME,ME) &&
             add(m0,NULL,w->mm0_b,merged,ME,1) && mm(res,m0,w->mm2_w,w->mm2_type,merged,ME,O) &&
             add(res,NULL,w->mm2_b,merged,O,2);
    } while (false);
    if (!gpu_end_commands()) ok = false;
    if (ok) ok = gpu_tensor_read(res,0,out,(uint64_t)merged*O*4);
    return ok;
}

static void take_error(char *err, size_t errlen, const char *fallback) {
    if (err && errlen) snprintf(err, errlen, "%s", g.error[0] ? g.error : fallback);
}

static void release(void) {
    free_buffers();
    if (g.tmp) (void)cudaFree(g.tmp);
    if (g.weights) (void)cudaFree(g.weights);
    if (g.cublas) (void)cublasDestroy(g.cublas);
    memset(&g, 0, sizeof(g));
}

static int upload_weights(const uint8_t *file, uint64_t data_pos, uint64_t size) {
    const uint64_t bytes = size - data_pos;
    void *device = NULL;
    if (!cuda_ok(cudaMalloc(&device, (size_t)bytes), "vision weights alloc")) return 0;
    g.weights = (char *)device;
    g.data_pos = data_pos;
    g.size = size;
    for (uint64_t copied = 0; copied < bytes; copied += WEIGHT_COPY_CHUNK) {
        const uint64_t n = std::min(WEIGHT_COPY_CHUNK, bytes - copied);
        if (!cuda_ok(cudaMemcpy(g.weights + copied, file + data_pos + copied, (size_t)n,
                                cudaMemcpyHostToDevice), "vision weights copy")) return 0;
    }
    return 1;
}

extern "C" int renza_vision_gpu_open(const uint8_t *file, uint64_t data_pos, uint64_t size,
        const renza_qwen4_vision_weights *w, uint32_t max_patches, char *err, size_t errlen) {
    if (g.open) {
        if (err && errlen) snprintf(err, errlen, "a vision encoder is already open in this process");
        return 0;
    }
    if (data_pos >= size) {
        if (err && errlen) snprintf(err, errlen, "the vision model has no tensor data");
        return 0;
    }
    const bool ok = cuda_ok(cudaSetDevice(0), "set device") &&
                    cublas_ok(cublasCreate(&g.cublas), "create handle") &&
                    cublas_ok(cublasSetMathMode(g.cublas, CUBLAS_TF32_TENSOR_OP_MATH), "set math mode") &&
                    upload_weights(file, data_pos, size) && ensure_buffers(max_patches, w);
    if (!ok) {
        take_error(err, errlen, "vision encoder GPU setup failed");
        release();
        return 0;
    }
    g.open = true;
    return 1;
}

extern "C" int renza_vision_gpu_encode(float *out, const float *patches, const float *pos_embed,
        uint32_t N, uint32_t grid_w, const renza_qwen4_vision_weights *w, char *err, size_t errlen) {
    g.error[0] = '\0';
    if (vision_encode(out, patches, pos_embed, N, grid_w, w)) return 1;
    take_error(err, errlen, "Qwen3.8 vision inference failed");
    return 0;
}

extern "C" void renza_vision_gpu_close(void) {
    if (!g.open) return;
    (void)cudaDeviceSynchronize();
    release();
}

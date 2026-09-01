#pragma once

// =====================================================================================
//  indexer_bf16_paged_mqa_logits — CUDA implementation
//
//  v2: the transfer is the bottleneck, so attack the transfer.
//
//  v1 measured 9-11 GB/s of PCIe on the *whole* kv_cache and nothing else; the kernel
//  was invisible. Three changes here:
//
//    1. compaction — kv_cache is sized for max_model_len (128 pages/batch) but only
//       ceil(context_len/block_size) pages are ever referenced (16/128 on TC1,
//       64/128 on TC2). We gather just those pages into a compact pool and rewrite the
//       block table to point into it, so the kernel is unchanged but 8x / 2x less data
//       crosses the bus.
//    2. pinned staging — the gather destination is cudaHostAlloc'd, which roughly
//       doubles H2D bandwidth versus the driver's internal staging of pageable memory.
//    3. pipelining — batches are cut into chunks; the CPU gather of chunk k+1 runs
//       while chunk k's H2D, kernel and D2H are in flight on its own stream.
//
//  Kernel core is unchanged from v1: one block per (bn, page), K page gathered once
//  into shared memory transposed to [d][t] and reused by all 64 heads.
// =====================================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <omp.h>

#include "ref_mqa_logits.h"
#include "allocator.h"
#include "Tensor.h"
#include "utils.h"

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t err_ = (expr);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                      << ": " << cudaGetErrorString(err_) << std::endl;          \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

namespace mqa {

constexpr int kBlockSize = 64;   // KV tokens per page
constexpr int kDim       = 128;  // head dim
constexpr int kNumHeads  = 64;
constexpr int kThreads   = 256;
constexpr int kGroups    = kThreads / kBlockSize;    // 4 head groups
constexpr int kHeadsPerGroup = kNumHeads / kGroups;  // 16 heads each
constexpr int kStreams   = 4;

// Phase timers, enabled with MQA_PROF=1. Diagnostic only; nothing is timed when off.
struct Prof {
    double plan = 0, gather = 0, issue = 0, sync = 0, out = 0;
    long n = 0;
    bool on = getenv("MQA_PROF") != nullptr;
    void tick() {
        if (!on || ++n % 100) return;
        fprintf(stderr, "[prof] plan %.3f | gather %.3f | issue %.3f | sync %.3f | out %.3f (ms/call)%c",
                plan / 100e3, gather / 100e3, issue / 100e3, sync / 100e3, out / 100e3, 10);
        plan = gather = issue = sync = out = 0;
    }
};
inline Prof &prof() { static Prof p; return p; }

// One CUDA block owns one (query row, KV page) pair.
//   blockIdx.x = page index inside the batch's logical KV sequence
//   blockIdx.y = query row, offset by bn_base (chunked launch)
// The page's K is staged once into shared memory, transposed to [dim][token] so that a
// warp (32 consecutive tokens, same dim) touches 32 consecutive halfwords instead of
// striding 256 B and serialising on one bank. All 64 heads then reuse it.
__global__ __launch_bounds__(kThreads) void mqa_logits_v2(
        const __nv_bfloat16 *__restrict__ q,        // [B, next_n, 64, 128]
        const __nv_bfloat16 *__restrict__ kv,       // compacted page pool
        const int32_t *__restrict__ block_tables,   // [B, max_num_blocks] -> compact id
        const int32_t *__restrict__ context_lens,   // [B]
        const float *__restrict__ weights,          // [B * next_n, 64]
        float *__restrict__ logits,                 // [B * next_n, max_model_len]
        int bn_base, int next_n, int max_model_len, int max_num_blocks)
{
    const int tid  = threadIdx.x;
    const int t    = tid & (kBlockSize - 1);   // token inside the page
    const int grp  = tid / kBlockSize;         // head group
    const int page = blockIdx.x;
    const int bn   = bn_base + blockIdx.y;

    const int batch_idx = bn / next_n;
    const int n         = bn - batch_idx * next_n;
    const int ctx       = context_lens[batch_idx];
    const int t_limit   = ctx - next_n + n;    // last finite token for this query row
    const int t_base    = page * kBlockSize;

    float *out_row = logits + (size_t)bn * max_model_len;

    // Wholly masked tile: just stamp -inf and leave. No KV touched.
    if (t_base > t_limit || page >= max_num_blocks) {
        if (tid < kBlockSize) {
            const int tg = t_base + tid;
            if (tg < max_model_len) out_row[tg] = -INFINITY;
        }
        return;
    }

    __shared__ __nv_bfloat16 sQ[kNumHeads * kDim];   // 16 KiB
    __shared__ __nv_bfloat16 sKT[kDim * kBlockSize]; // 16 KiB, [d][t]
    __shared__ float sW[kNumHeads];                  // 256 B
    __shared__ float sPart[kGroups][kBlockSize];     // 1 KiB

    // ---- stage Q (all 64 heads of this query row) and the head weights ----
    {
        const uint4 *src = reinterpret_cast<const uint4 *>(q + (size_t)bn * kNumHeads * kDim);
        uint4 *dst = reinterpret_cast<uint4 *>(sQ);
        #pragma unroll
        for (int i = tid; i < (kNumHeads * kDim) / 8; i += kThreads) dst[i] = src[i];
        if (tid < kNumHeads) sW[tid] = weights[(size_t)bn * kNumHeads + tid];
    }

    // ---- stage this page's K, transposed into [d][t] ----
    // block_size == page size, so one table lookup covers the whole tile.
    {
        const int32_t phys = block_tables[(size_t)batch_idx * max_num_blocks + page];
        const uint4 *src = reinterpret_cast<const uint4 *>(kv + (size_t)phys * kBlockSize * kDim);
        #pragma unroll
        for (int i = tid; i < (kBlockSize * kDim) / 8; i += kThreads) {
            const int tok = i / (kDim / 8);
            const int d0  = (i % (kDim / 8)) * 8;
            const uint4 v = src[i];
            const __nv_bfloat16 *vb = reinterpret_cast<const __nv_bfloat16 *>(&v);
            #pragma unroll
            for (int j = 0; j < 8; ++j) sKT[(d0 + j) * kBlockSize + tok] = vb[j];
        }
    }
    __syncthreads();

    // ---- each thread: one token, 16 heads. relu(q.k) * w, summed over its heads ----
    float partial = 0.f;
    const bool live = (t_base + t) <= t_limit && (t_base + t) < ctx;
    if (live) {
        const int h0 = grp * kHeadsPerGroup;
        for (int hh = 0; hh < kHeadsPerGroup; ++hh) {
            const __nv_bfloat16 *qh = sQ + (size_t)(h0 + hh) * kDim;
            float acc = 0.f;
            #pragma unroll
            for (int d = 0; d < kDim; ++d)
                acc = fmaf(__bfloat162float(qh[d]), __bfloat162float(sKT[d * kBlockSize + t]), acc);
            partial = fmaf(fmaxf(acc, 0.f), sW[h0 + hh], partial);
        }
    }

    // ---- reduce the 4 head groups and write the tile ----
    sPart[grp][t] = partial;
    __syncthreads();
    if (tid < kBlockSize) {
        const int tg = t_base + tid;
        if (tg < max_model_len) {
            float s = sPart[0][tid] + sPart[1][tid] + sPart[2][tid] + sPart[3][tid];
            out_row[tg] = (tg <= t_limit && tg < ctx) ? s : -INFINITY;
        }
    }
}

// -------------------------------------------------------------------------------------
//  Host side: persistent device + pinned staging buffers, grown on demand and reused
//  across the 100 timed runs (allocation is not part of the algorithm).
// -------------------------------------------------------------------------------------
struct Ctx {
    void *d_kv = nullptr, *d_q = nullptr, *d_bt = nullptr, *d_cl = nullptr,
         *d_w = nullptr, *d_out = nullptr;
    size_t c_kv = 0, c_q = 0, c_bt = 0, c_cl = 0, c_w = 0, c_out = 0;

    void *h_kv = nullptr, *h_q = nullptr, *h_bt = nullptr, *h_cl = nullptr,
         *h_w = nullptr, *h_out = nullptr;
    size_t p_kv = 0, p_q = 0, p_bt = 0, p_cl = 0, p_w = 0, p_out = 0;

    std::vector<int32_t> src_pages;   // compact id -> physical page in the caller's cache
    std::vector<int64_t> page_begin;  // per batch, first compact id

    cudaStream_t stream[kStreams] = {};
    cudaEvent_t  meta_ready = nullptr;
    bool init = false;

    static void ensure_dev(void **p, size_t &cap, size_t need) {
        if (cap >= need) return;
        if (*p) CUDA_CHECK(cudaFree(*p));
        need = need + need / 4 + 256;
        CUDA_CHECK(cudaMalloc(p, need));
        cap = need;
    }
    static void ensure_pin(void **p, size_t &cap, size_t need) {
        if (cap >= need) return;
        if (*p) CUDA_CHECK(cudaFreeHost(*p));
        need = need + need / 4 + 256;
        CUDA_CHECK(cudaHostAlloc(p, need, cudaHostAllocDefault));
        cap = need;
    }
    void boot() {
        if (init) return;
        for (int i = 0; i < kStreams; ++i) CUDA_CHECK(cudaStreamCreate(&stream[i]));
        CUDA_CHECK(cudaEventCreateWithFlags(&meta_ready, cudaEventDisableTiming));
        init = true;
    }
};

inline Ctx &ctx_of() { static Ctx c; return c; }

// Multi-threaded memcpy: the pageable<->pinned hops are wide enough that a single core
// (~10 GB/s) would become the new bottleneck.
inline void par_memcpy(void *dst, const void *src, size_t bytes)
{
    constexpr size_t kGrain = 1 << 20;
    const int64_t nchunk = (int64_t)((bytes + kGrain - 1) / kGrain);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < nchunk; ++i) {
        const size_t off = (size_t)i * kGrain;
        const size_t len = bytes - off < kGrain ? bytes - off : kGrain;
        memcpy((char *)dst + off, (const char *)src + off, len);
    }
}

}  // namespace mqa

inline void indexer_bf16_paged_mqa_logits(
    const Tensor<bfloat16_t, 4> &q,          // [batch_size, next_n, num_heads, dim]
    const Tensor<bfloat16_t, 4> &kv_cache,   // [num_blocks, block_size, 1, dim]
    const Tensor<int64_t, 2> &block_tables,  // [batch_size, max_num_blocks]
    const Tensor<int64_t, 1> &context_lens,  // [batch_size]
    const Tensor<float, 2> &weights,         // [batch_size * next_n, num_heads]
    const Tensor<float, 2> &output,          // [batch_size * next_n, max_model_len]
    int64_t batch_size,
    int64_t next_n,
    int64_t num_heads,
    int64_t dim,
    int64_t block_size,
    int64_t max_model_len)
{
    // Shapes this kernel is specialised for; anything else falls back to the CPU ref.
    if (num_heads != mqa::kNumHeads || dim != mqa::kDim || block_size != mqa::kBlockSize) {
        ref_bf16_paged_mqa_logits<float>(q, kv_cache, block_tables, context_lens, weights,
            output, batch_size, next_n, num_heads, dim, block_size, max_model_len);
        return;
    }

    auto &c = mqa::ctx_of();
    c.boot();

    const int64_t B   = batch_size;
    const int64_t mnb = block_tables.sizes()[1];
    const int64_t bn_total = B * next_n;
    const size_t page_bytes = (size_t)mqa::kBlockSize * mqa::kDim * sizeof(bfloat16_t);
    auto &pf = mqa::prof();
    double t0 = get_clock_us(), t_gather = 0;

    // ---- plan the compaction: which physical pages are actually referenced ----
    // kv_cache is sized for max_model_len; a batch only touches ceil(ctx/block_size)
    // pages. Renumber those into a dense pool and rewrite the block table to match.
    const size_t bt_bytes = (size_t)B * mnb * sizeof(int32_t);
    mqa::Ctx::ensure_pin(&c.h_bt, c.p_bt, bt_bytes);
    mqa::Ctx::ensure_pin(&c.h_cl, c.p_cl, (size_t)B * sizeof(int32_t));
    int32_t *hbt = (int32_t *)c.h_bt;
    int32_t *hcl = (int32_t *)c.h_cl;
    memset(hbt, 0, bt_bytes);

    const int64_t *bt_src = block_tables.data_ptr();
    const int64_t *cl_src = context_lens.data_ptr();

    c.page_begin.resize(B + 1);
    c.src_pages.clear();
    int64_t total_pages = 0;
    for (int64_t b = 0; b < B; ++b) {
        c.page_begin[b] = total_pages;
        const int32_t ctx = (int32_t)cl_src[b];
        hcl[b] = ctx;
        int64_t np = ceil_div<int64_t>(ctx, block_size);
        if (np > mnb) np = mnb;
        for (int64_t p = 0; p < np; ++p) {
            c.src_pages.push_back((int32_t)bt_src[b * mnb + p]);
            hbt[b * mnb + p] = (int32_t)total_pages++;
        }
    }
    c.page_begin[B] = total_pages;

    // ---- buffers ----
    const size_t q_bytes = (size_t)q.numel() * sizeof(bfloat16_t);
    const size_t w_bytes = (size_t)weights.numel() * sizeof(float);
    const size_t o_bytes = (size_t)output.numel() * sizeof(float);
    const size_t kv_bytes = (size_t)total_pages * page_bytes;

    mqa::Ctx::ensure_pin(&c.h_q, c.p_q, q_bytes);
    mqa::Ctx::ensure_pin(&c.h_w, c.p_w, w_bytes);
    mqa::Ctx::ensure_pin(&c.h_out, c.p_out, o_bytes);
    // Worst-case capacity, so nothing regrows mid-run: a testcase cycles through several
    // datasets whose context_lens differ, and a cudaHostAlloc/cudaFree pair inside a timed
    // call costs far more than the transfer it serves.
    const size_t kv_cap = (size_t)B * mnb * page_bytes;
    mqa::Ctx::ensure_pin(&c.h_kv, c.p_kv, kv_cap);
    mqa::Ctx::ensure_dev(&c.d_q, c.c_q, q_bytes);
    mqa::Ctx::ensure_dev(&c.d_w, c.c_w, w_bytes);
    mqa::Ctx::ensure_dev(&c.d_out, c.c_out, o_bytes);
    mqa::Ctx::ensure_dev(&c.d_kv, c.c_kv, kv_cap);
    mqa::Ctx::ensure_dev(&c.d_bt, c.c_bt, bt_bytes);
    mqa::Ctx::ensure_dev(&c.d_cl, c.c_cl, (size_t)B * sizeof(int32_t));

    pf.plan += get_clock_us() - t0;

    // ---- metadata + Q up front; every stream waits on it once ----
    mqa::par_memcpy(c.h_q, q.data_ptr(), q_bytes);
    memcpy(c.h_w, weights.data_ptr(), w_bytes);
    CUDA_CHECK(cudaMemcpyAsync(c.d_q, c.h_q, q_bytes, cudaMemcpyHostToDevice, c.stream[0]));
    CUDA_CHECK(cudaMemcpyAsync(c.d_w, c.h_w, w_bytes, cudaMemcpyHostToDevice, c.stream[0]));
    CUDA_CHECK(cudaMemcpyAsync(c.d_bt, c.h_bt, bt_bytes, cudaMemcpyHostToDevice, c.stream[0]));
    CUDA_CHECK(cudaMemcpyAsync(c.d_cl, c.h_cl, (size_t)B * sizeof(int32_t),
                               cudaMemcpyHostToDevice, c.stream[0]));
    CUDA_CHECK(cudaEventRecord(c.meta_ready, c.stream[0]));
    for (int i = 1; i < mqa::kStreams; ++i)
        CUDA_CHECK(cudaStreamWaitEvent(c.stream[i], c.meta_ready, 0));

    // ---- pipeline over batch chunks: gather(k+1) || H2D(k) || kernel(k) || D2H(k) ----
    const unsigned pages_x = (unsigned)ceil_div<int64_t>(max_model_len, block_size);
    const size_t row_bytes = (size_t)max_model_len * sizeof(float);
    const int32_t *src_pages = c.src_pages.data();
    const char *kv_base = (const char *)kv_cache.data_ptr();

    t0 = get_clock_us();
    // Aim for ~2 chunks in flight per stream: enough to hide a gather behind the previous
    // chunk's H2D + kernel, without paying launch overhead on tiny tiles.
    int64_t chunk_pages = ceil_div<int64_t>(total_pages, mqa::kStreams * 2);
    if (chunk_pages < 32) chunk_pages = 32;

    int si = 0;
    for (int64_t b0 = 0; b0 < B;) {
        int64_t b1 = b0;
        while (b1 < B && c.page_begin[b1] - c.page_begin[b0] < chunk_pages) ++b1;
        if (b1 == b0) b1 = b0 + 1;

        cudaStream_t s = c.stream[si];
        si = (si + 1) % mqa::kStreams;

        const int64_t p0 = c.page_begin[b0], p1 = c.page_begin[b1];
        const double tg = get_clock_us();
        #pragma omp parallel for schedule(static)
        for (int64_t i = p0; i < p1; ++i)
            memcpy((char *)c.h_kv + (size_t)i * page_bytes,
                   kv_base + (size_t)src_pages[i] * page_bytes, page_bytes);

        t_gather += get_clock_us() - tg;

        if (p1 > p0)
            CUDA_CHECK(cudaMemcpyAsync((char *)c.d_kv + (size_t)p0 * page_bytes,
                                       (char *)c.h_kv + (size_t)p0 * page_bytes,
                                       (size_t)(p1 - p0) * page_bytes,
                                       cudaMemcpyHostToDevice, s));

        const int64_t bn0 = b0 * next_n, bnc = (b1 - b0) * next_n;
        mqa::mqa_logits_v2<<<dim3(pages_x, (unsigned)bnc), mqa::kThreads, 0, s>>>(
            (const __nv_bfloat16 *)c.d_q, (const __nv_bfloat16 *)c.d_kv,
            (const int32_t *)c.d_bt, (const int32_t *)c.d_cl,
            (const float *)c.d_w, (float *)c.d_out,
            (int)bn0, (int)next_n, (int)max_model_len, (int)mnb);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync((char *)c.h_out + (size_t)bn0 * row_bytes,
                                   (char *)c.d_out + (size_t)bn0 * row_bytes,
                                   (size_t)bnc * row_bytes, cudaMemcpyDeviceToHost, s));
        b0 = b1;
    }

    pf.gather += t_gather;
    pf.issue += get_clock_us() - t0 - t_gather;
    t0 = get_clock_us();
    CUDA_CHECK(cudaDeviceSynchronize());
    pf.sync += get_clock_us() - t0;
    t0 = get_clock_us();
    mqa::par_memcpy(output.data_ptr(), c.h_out, o_bytes);
    pf.out += get_clock_us() - t0;
    pf.tick();
    (void)bn_total;
}

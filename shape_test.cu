// Shape sweep for the bonus criterion. Same driver logic as main.cu (which is untouched),
// but it walks a list of num_heads / dim / block_size / next_n combinations and checks each
// against the double-precision CPU reference.  Build: see build_shape_test.sh
#include <iomanip>
#include <cstring>
#include <vector>
#include <iostream>
#include <random>

#include "Tensor.h"
#include "utils.h"
#include "allocator.h"
#include "testcase.h"
#include "ref_mqa_logits.h"
#include "indexer_mqa_logits.h"

constexpr int64_t MAX_ALLOC_SIZE = 1200LL * 1024 * 1024;

struct Shape { int64_t heads, dim, bs, batch, next_n, mml, avg_kv; const char *note; };

static const Shape kShapes[] = {
    { 64, 128,  64, 16, 2, 2048, 1024, "graded shape (fast path)" },
    { 32, 128,  64, 16, 2, 2048, 1024, "fewer heads" },
    {128, 128,  64,  8, 1, 2048, 1024, "more heads" },
    {  1, 128,  64, 16, 1, 2048, 1024, "single head" },
    { 64,  32,  64, 16, 1, 2048, 1024, "dim 32 (1 elem/lane)" },
    { 64,  64,  64, 16, 2, 2048, 1024, "dim 64" },
    { 32, 256,  64, 16, 1, 2048, 1024, "dim 256" },
    { 16, 512,  64,  8, 1, 2048, 1024, "dim 512 (16 elem/lane, max)" },
    { 40,  96,  48, 16, 2, 2016,  900, "dim 96, block 48 (non-pow2)" },
    { 64, 128,  16, 16, 1, 2048, 1024, "block 16" },
    { 32, 128, 128, 16, 3, 2048, 1024, "block 128, next_n 3" },
    { 64, 128,  64, 16, 4, 2048,  700, "next_n 4" },
    { 64, 100,  64,  8, 1, 2048, 1024, "dim 100 -> CPU ref fallback" },
};

static bool run_shape(const Shape &sh)
{
    TestCaseParams p;
    p.batch_size = sh.batch; p.next_n = sh.next_n; p.num_heads = sh.heads;
    p.dim = sh.dim; p.block_size = sh.bs; p.max_model_len = sh.mml;
    p.check_correctness = true; p.num_runs = 4; p.avg_kv_len = sh.avg_kv;

    std::mt19937 rng(12345);
    Allocator allocator(MAX_ALLOC_SIZE);
    auto datalist = generate_datalist(p, allocator, rng);

    for (int64_t i = 0; i < p.num_runs; ++i) {
        auto &d = datalist[i % datalist.size()];
        indexer_bf16_paged_mqa_logits(d.q, d.kv_cache, d.block_tables, d.context_lens,
            d.weights, d.logits, p.batch_size, p.next_n, p.num_heads, p.dim, p.block_size,
            p.max_model_len);
    }
    auto &d = datalist[0];
    ref_bf16_paged_mqa_logits<double>(d.q, d.kv_cache, d.block_tables, d.context_lens,
        d.weights, d.ref_logits, p.batch_size, p.next_n, p.num_heads, p.dim, p.block_size,
        p.max_model_len);
    bool mask_ok = check_mask_and_replace(d.logits, d.ref_logits);
    double cd = calc_cos_diff(d.logits, d.ref_logits);
    bool ok = mask_ok && cd < 5e-6;
    std::cout << (ok ? "  PASS  " : "  FAIL  ")
              << "heads=" << std::setw(3) << p.num_heads
              << " dim=" << std::setw(4) << p.dim
              << " block=" << std::setw(4) << p.block_size
              << " next_n=" << p.next_n
              << " mask=" << (mask_ok ? "ok" : "BAD")
              << std::scientific << std::setprecision(3) << " cos_diff=" << cd
              << "   (" << sh.note << ")" << std::endl;
    return ok;
}

int main()
{
    int bad = 0;
    for (const auto &sh : kShapes) if (!run_shape(sh)) ++bad;
    std::cout << (bad ? "SHAPE SWEEP FAILED: " : "SHAPE SWEEP OK, failures: ") << bad << std::endl;
    return bad != 0;
}

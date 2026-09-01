# indexer_bf16_paged_mqa_logits 优化报告

只修改了 `indexer_mqa_logits.h`，其余原有文件（`main.cu` / `ref_mqa_logits.h` /
`testcase.h` / `Tensor.h` / `allocator.h` / `utils.h` / `run.sh` / `README.md`）保持原样。
另外新增了两个不参与评测的辅助文件：`build_local.sh`（本机原生架构构建）和
`shape_test.cu`（任意形状的正确性扫描）。

## 1. 结论

| 版本 | 关键改动 | TC1 GFLOPS | TC2 GFLOPS | cos_diff (TC1 / TC2) |
|---|---|---|---|---|
| CPU 参考实现 | 32 线程 OpenMP | 86.70 | 88.69 | 1.8e-14 / 4.4e-14 |
| v1 | 朴素 CUDA kernel，整块上传 KV | 144.94 | 342.22 | 2.3e-14 / 5.3e-14 |
| v2 | 页压缩 + pinned 中转 + 多流流水 | 554.18 | 525.69 | 2.3e-14 / 5.3e-14 |
| v3 | Tensor Core（BF16 m16n8k16 MMA） | 680.70 | 634.33 | 2.7e-14 / 5.9e-14 |
| v4 | 一个 block 吃 4 页 + 只启动能用到的页 | 694.10 | 633.46 | 2.7e-14 / 5.9e-14 |
| v5 | 切片 H2D + 常驻 OpenMP 并行域 | ~790 | ~755 | 2.7e-14 / 5.9e-14 |
| **v6（最终）** | **D2H 只回传活前缀** | **~920** | **~768** | **2.675637e-14 / 5.917489e-14** |

最终相对 CPU 参考实现：**TC1 约 10.6×，TC2 约 8.7×**。
正确性余量：要求 `cos_diff < 5e-6`，实测 2.7e-14 / 5.9e-14，低了八个数量级；
输出 mask 逐元素校验通过。

单次调用耗时：TC1 1.13 ms，TC2 11.17 ms（含全部 H2D/D2H，因为 `main.cu`
的计时器包住了整个算子调用）。

多次运行的抖动（同一二进制跑 8 次）：TC1 845–945，TC2 748–771。
抖动主要来自主机侧的 gather 和 CPU 频率，不是 kernel。
下文所有对比数字都在同一台机器、同一组配置下取得。

## 2. 环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop (Blackwell, sm_120, 12 GiB) |
| CPU | 32 逻辑核 |
| 系统 | WSL2 Ubuntu 24.04，驱动 610.88，CUDA 13.3 |
| 编译 | `nvcc -O3 -std=c++17 -Xcompiler -fopenmp`，同时生成 sm_86 / sm_120 |
| 运行 | `OMP_NUM_THREADS=8 ./main` |

`run.sh` 未做任何修改（其中写死 `-arch=sm_86`，在本机通过 PTX JIT 运行）；
`build_local.sh` 是我自己加的构建脚本，额外产出 sm_120 原生代码，避免每次启动 JIT。

## 3. 问题分析：这是一道 PCIe 题，不是一道算力题

先算两个数（TC2）：

- 计算量：`2 × total_ctx × next_n × num_heads × dim = 2 × 524288 × 1 × 64 × 128 ≈ 8.6 GFLOP`
- 必须搬运的数据：被 `block_tables` 引用的 KV 页 = `8192 页 × 64 × 128 × 2 B = 128 MiB`

关键在于：**所有输入张量都由 `allocator.h` 分配在主机 mmap 内存里**（既不是 pinned，
也不在显存中），而 `main.cu` 的计时器包住了整个算子调用，也就是说 H2D/D2H 全部计入耗时。

实测本机传输带宽：

| 通道 | 带宽 |
|---|---|
| H2D pinned | 14.2 GB/s |
| H2D pageable | 12.9 GB/s |
| D2H pinned | 14.1 GB/s |
| 显存内部拷贝 | 472 GB/s |
| 主机 gather（8 线程，随机页） | ~19 GB/s 有效（DRAM 实际流量 ~37 GB/s） |

于是 PCIe 上的"算术强度"只有 `8.6 GFLOP / 136 MB ≈ 63 flop/byte`，
乘以 14.2 GB/s 得到 **TC2 的理论天花板约 0.9 TFLOPS**；TC1 因为数据量小、
`next_n=2` 复用了同一份 KV，强度是 114 flop/byte，天花板约 1.6 TFLOPS。

结论：kernel 本身（Tensor Core 版本约 0.4–1.0 ms）远不是瓶颈，
**优化的主战场是"少搬、早搬、边搬边算"**。这也解释了为什么下面 v2 的收益（3.8×）
比 v3 上 Tensor Core 的收益（1.2×）大得多。

## 4. 优化路线

### v1：朴素 CUDA kernel

一个 block 负责一个 `(bn, page)`，把 K 页读进 shared memory，每个线程算一个
`(head, token)` 点积，`__shfl` 归约后写回。整块 `kv_cache` 原样上传。

但它很慢：TC1 只有 145 GFLOPS。因为 `kv_cache` 的物理大小是按
`max_model_len` 分配的（每 batch 128 页），而一次调用只会碰到
`ceil(ctx/64)` 页 —— TC1 是 16/128，TC2 是 64/128。也就是说 **87% / 50%
的 PCIe 流量是纯浪费**。

### v2：页压缩 + pinned 中转 + 多流（145 → 554 GFLOPS）

三件事一起做：

1. **页压缩**：扫一遍 `block_tables`，只把真正被引用的页按逻辑顺序 gather 到一个
   稠密页池里，同时把 `block_tables` 重写成指向页池的紧凑编号。TC1 上传量
   64 MiB → 8 MiB，TC2 256 MiB → 128 MiB。
2. **pinned 中转**：gather 的目的地是 `cudaHostAlloc` 的常驻缓冲区，一次 gather
   同时完成"去稀疏"和"进 pinned"，H2D 因此走 14.2 而不是 12.9 GB/s，且能异步。
3. **多流 + 分块**：按 batch 切块，`gather(k+1) ‖ H2D(k) ‖ kernel(k) ‖ D2H(k)`
   流水；Q/weights/block_tables 只上传一次，用一个 event 让所有流等它。

所有设备端和 pinned 缓冲区都按最坏情况一次性分配（`ensure_dev`/`ensure_pin` 只增不减）——
一个 testcase 会轮换 7~28 份数据集，如果在计时区间内 `cudaMalloc`，
单次分配的开销就比它服务的传输还大。

### v3：Tensor Core（554 → 681 GFLOPS）

每页的计算本质上是一个 `64(head) × 64(token) × 128(dim)` 的 GEMM：
`S[h][t] = Q[h][:] · K[t][:]`，正好是 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
的形状（sm_80+；Hopper 的 `wgmma` 在 sm_120 上不可用，所以用 m16n8k16）。

- Q 和 K 都以自然的 `[row][dim]` 布局进 shared memory，这个布局同时就是 mma 的
  A/B fragment 想要的布局，`ldmatrix` 之外不需要额外转置。
- 8 个 warp 把 `64×64` 输出切成 **4 个 head 组 × 2 个 token 半区**，每个 warp 独占
  `16(head) × 32(token)` 的一片，任何两个 warp 都不会写同一个 partial。
- `acc[4][4]` 全部留在寄存器里，`ks` 沿 dim 128 展开成 8 次 k=16 的 mma。

**Shared memory 的 bank 冲突**：行距取 `kSmemStride = 136` 个 bf16 而不是 128。
136 bf16 = 68 个 32-bit word，`68 % 32 == 4`，于是一个 fragment 的 8 行落在 8 个
不同的 bank 上；如果用 128（= 64 word，`64 % 32 == 0`）则 8 行全撞同一个 bank，
变成 8-way 冲突。代价是每块多 1 KiB shared memory。

Q/K/W/partial 合计约 36 KB，在 48 KB 静态 shared memory 上限内，不需要
`cudaFuncSetAttribute` 开动态 smem。

### v4：一个 block 吃 4 页 + 只启动能用到的页（681 → 694 GFLOPS）

- **`kPagesPerBlock = 4`**：Q 的 staging（17 KiB）在一个 block 内被 4 页复用，
  block 数量降到 1/4，Q 的 shared memory 装载开销摊薄 4 倍。
- **`pages_used`**：kernel 的 grid 只覆盖 `ceil(本 chunk 内最大 ctx / 64)` 页。
  剩下的行尾全是 `-inf`，不值得为它调度几千个只写 256 B 的 block。
- 全行被 mask 掉的 block 直接早退。

### v5：切片 H2D + 常驻 OpenMP 并行域（694 → ~790 GFLOPS）

这一步的关键认识是：**gather 的粒度和 kernel 启动的粒度应该解耦**。

之前"按 batch 分块"意味着 chunk 既是 gather+H2D 的单位，又是 kernel 的单位。
但这两者的最优粒度完全不同：

- H2D 想要**细**：只要第一片 gather 完就该让 DMA 引擎动起来，别等整个 batch；
- kernel 想要**粗**：`pages_used` 是 chunk 内的最大值，切细了会产生大量空转 block，
  再加上每次启动的开销。

于是引入 `MQA_GCHUNK`（gather/H2D 的页粒度）独立于 `MQA_CHUNK`（kernel 粒度），
后者默认取"整批一次启动"。`MQA_GCHUNK` 自适应取 `clamp(total_pages/24, 64, 512)`：
目标约 24 片，既让第一次 DMA 尽早发出，又不让单片小于 1 MiB（否则
WSL2 上每次 `cudaMemcpyAsync` 约 15 µs 的 API 开销开始占主导）。

第二个改动：**整次调用只开一个 `#pragma omp parallel` 区域**，每片 KV 用
`#pragma omp for` 做 worksharing，`#pragma omp master` 在该片的隐式 barrier 一过
就立刻发出它的 H2D。这样片 k 的 DMA 与片 k+1 的 gather 重叠，却不必为每片
付一次 OpenMP fork/join（实测约 15 µs/片；细粒度切片时 TC2 的 gather 会从
8.2 ms 涨到 13.4 ms，就是被 fork/join 吃掉的）。

### v6：D2H 只回传"活的前缀"（~790 → ~940 / 770 GFLOPS）

输出是 `[batch*next_n, max_model_len=8192]`，但一行里只有前
`pages_used × block_size` 列可能不是 `-inf`：TC1 大约 20%，TC2 大约 70%。

所以把 D2H 换成 `cudaMemcpy2DAsync`，只回传这段前缀；`-inf` 的尾巴由主机在
"把结果从 pinned 缓冲拷回 `output`"这一步顺手写进去 —— 主机写的字节数一样多
（甚至更省，因为尾巴不需要读源），却省下 TC1 约 1.6 MiB / TC2 约 1.1 MiB 的 PCIe 流量，
连设备端的 `fill_inf` kernel 也一并省掉。

TC1 因此从 808 → 942 GFLOPS（+17%），`out` 阶段 0.092 → 0.058 ms。
这是整条路线上"性价比"最高的一步：改动约 20 行。

## 5. 阶段耗时剖析

`MQA_PROF=1` 会在算子内部按阶段累计耗时（每个 testcase 独立计数，丢掉前 10 次
warm-up，因为那几次要付一次性的 `cudaMalloc`/`cudaHostAlloc`，`main.cu` 自己也把
第 0 次排除在平均之外）。最终版本（ms/call，90 次平均）：

| 阶段 | 含义 | TC1 | TC2 |
|---|---|---|---|
| plan | 扫 `block_tables` 建压缩计划、检查缓冲区容量 | 0.003 | 0.017 |
| meta | Q/weights/bt/cl 进 pinned 并发出上传、event 同步 | 0.069 | 0.168 |
| gather | 主机把散页拷进 pinned 页池（与 DMA 重叠） | 0.265 | 7.818 |
| issue | 发起 kernel 与各次 copy 的 API 开销 | 0.113 | 0.500 |
| sync | `cudaDeviceSynchronize`，等剩余 DMA/kernel | 0.622 | 2.457 |
| out | 从 pinned 拷回 `output` 并写 `-inf` 尾巴 | 0.058 | 0.207 |
| **合计** | | **1.130** | **11.167** |

合计与 `main.cu` 测出的单次耗时基本吻合（阶段划分没有黑洞）。

**怎么读这张表**：TC2 的 `gather` 7.8 ms 看着吓人，但它是在 master 线程发出 DMA
之前记的，与 9.6 ms 的 H2D 完全重叠 —— 证据是 `gather + sync = 10.3 ms` 而不是
`7.8 + 9.6 = 17.4 ms`。真正串行残留的是最后一片 H2D 之后的 kernel + D2H。

### 上界分析

| | TC1 | TC2 |
|---|---|---|
| 压缩后 KV | 8 MiB | 128 MiB |
| Q | 1 MiB | 2 MiB |
| D2H（活前缀） | ~0.45 MiB | ~2.9 MiB |
| H2D 下限 @14.2 GB/s | 0.665 ms | 9.60 ms |
| 实测 | 1.130 ms | 11.167 ms |
| 距 PCIe roofline | 59% | **86%** |

TC2 已经贴到 PCIe 屋顶（剩下的 1.5 ms 是最后一片 KV 到齐之后才能跑的 kernel
约 0.83 ms，加上 D2H 和主机回写）。TC1 数据量小，per-call 固定开销
（meta + issue ≈ 0.18 ms）和不可重叠的 kernel 尾巴占比更高。

要吃掉 TC1 剩下的 40%，唯一的办法是让 kernel 与 H2D 重叠（把 kernel 也切块，
或者用 event 让 kernel 在独立流上等每片 H2D）。我实测了前者，反而更慢
（见下面的负结果），因为把 kernel 插进拷贝流会打断 copy engine 的连续性，
而 copy engine 才是瓶颈。这条路我保留为未完成项，而不是假装它不存在。

## 6. 拓展加分：任意 num_heads / dim / block_size

Tensor Core kernel 是针对 `num_heads=64, dim=128, block_size=64` 特化的
（fragment 形状、shared memory 预算、warp 分工都写死了）。为了不在其他形状上
退化成 CPU 参考实现，另写了一个 `mqa_logits_generic`：

- **一个 warp 负责一个 token**：先把该 token 的 K 向量按 `dim/32` 个元素每 lane
  读进寄存器（lane 跨步访问，保证 coalesced），然后沿 head 循环。K 因此在整个
  head 维度上只从 DRAM 读一次，Q 反复读但一直命中 L1/L2。
- ReLU 在 head 内、权重相乘之前，所以每个 head 都必须先做完整的点积 —— 每个
  head 一次 5 步 `__shfl_xor` warp 归约，这是它比特化版慢的主要原因。
- 寄存器数组用 `#pragma unroll` + `if (i < dpl)` 展开成静态下标，避免落到 local memory；
  上限 `kMaxDimPerLane = 16`，即支持 `dim ≤ 512`。
- 主机侧的页压缩、pinned 中转、切片流水、活前缀 D2H 全部复用，与形状无关。

约束：`dim % 32 == 0 && dim <= 512`。超出这个范围（例如 `dim = 100`）才回落到 CPU 参考实现。

验证用 `shape_test.cu`（我自己加的文件，不修改任何原有文件），逐形状与
`ref_bf16_paged_mqa_logits<double>` 对比，13 组全部通过：

```
PASS heads= 64 dim= 128 block=  64 next_n=2  cos_diff=2.898e-14   (graded shape, fast path)
PASS heads= 32 dim= 128 block=  64 next_n=2  cos_diff=8.327e-15   (fewer heads)
PASS heads=128 dim= 128 block=  64 next_n=1  cos_diff=1.144e-14   (more heads)
PASS heads=  1 dim= 128 block=  64 next_n=1  cos_diff=3.886e-15   (single head)
PASS heads= 64 dim=  32 block=  64 next_n=1  cos_diff=4.219e-15   (dim 32, 1 elem/lane)
PASS heads= 64 dim=  64 block=  64 next_n=2  cos_diff=1.021e-14   (dim 64)
PASS heads= 32 dim= 256 block=  64 next_n=1  cos_diff=6.328e-15   (dim 256)
PASS heads= 16 dim= 512 block=  64 next_n=1  cos_diff=6.217e-15   (dim 512, 16 elem/lane)
PASS heads= 40 dim=  96 block=  48 next_n=2  cos_diff=9.548e-15   (dim 96, block 48, 非 2 的幂)
PASS heads= 64 dim= 128 block=  16 next_n=1  cos_diff=9.215e-15   (block 16)
PASS heads= 32 dim= 128 block= 128 next_n=3  cos_diff=7.994e-15   (block 128, next_n 3)
PASS heads= 64 dim= 128 block=  64 next_n=4  cos_diff=3.231e-14   (next_n 4)
PASS heads= 64 dim= 100 block=  64 next_n=1  cos_diff=1.055e-14   (dim 100 -> CPU ref 回落)
SHAPE SWEEP OK, failures: 0
```

另外 `MQA_FORCE_GENERIC=1` 可以把评测形状也强制走通用 kernel，
用官方 harness 自己的 `cos_diff` 校验它：TC1 380 GFLOPS / TC2 308 GFLOPS，
`cos_diff` 1.2e-14 / 3.2e-14 —— 仍然是 CPU 参考实现的 3.5 倍，但只有特化版的 40%。
两者共存是有意的：形状对上就走 Tensor Core，对不上也不至于掉回 CPU。

## 7. 常驻 KV Cache 模式（`MQA_RESIDENT_KV=1`，单独标注，不作为主指标）

评测 harness 每次调用都把 KV cache 交在主机 mmap 内存里，所以上面所有数字都
包含"每次调用重新搬 128 MiB 过 PCIe"。真实的 vLLM 不是这样工作的：KV cache
常驻显存，只有新写入的 token 才会动。

为了量化"如果 KV 已经在显存里会怎样"，加了一个可选模式：按
`(kv_cache 指针, block_tables 指纹, context_lens)` 三元组缓存显存页池，
数据集重复出现时直接跳过 gather 和 H2D。`main.cu` 会在 100 次调用里轮换
7~28 份数据集，所以命中率很高。

| | TC1 | TC2 |
|---|---|---|
| 默认（每次重传） | ~920 GFLOPS | ~768 GFLOPS |
| `MQA_RESIDENT_KV=1` | **1231 GFLOPS** | **4447 GFLOPS** |

阶段耗时（resident）：TC1 `plan 0.105 | meta 0.078 | gather 0.083 | issue 0.050 | sync 0.381 | out 0.070`；
TC2 `plan 0.016 | meta 0.075 | gather 0.000 | issue 0.043 | sync 0.834 | out 0.146`。
TC2 的 4447 GFLOPS 就是这个 kernel 在"没有 PCIe 税"时的真实水平，
`sync 0.834 ms` 基本就是 kernel 本身的耗时。

**我不把它当作成绩**：题目给的数据在主机内存里，跳过传输就不是在解同一道题。
默认关闭，写在这里是因为它回答了"kernel 到底有多快"这个问题，
也说明剩下的优化空间在传输而不在计算。

## 8. 试过但没有收益的方向（负结果）

WSL2 下 `ncu` 拿不到硬件计数器（`ERR_NVGPUCTRPERM`，要改注册表 + 重启），
所以下面的判断全部来自主机侧阶段计时和针对性的微基准，而不是 profiler。

1. **`cudaMemcpyBatchAsync`（CUDA 13 的批量 scatter-gather）**：本想用它一次提交
   8192 个页拷贝，直接从调用方 mmap 内存散读，省掉主机 gather。
   源是 `cudaHostRegister` 过的 mmap 时必然 `illegal memory access`（即使 count=64），
   换成 `cudaHostAlloc` 的源就正常；补 `srcLocHint/dstLocHint` 也无效。
   判定为该驱动上的实现问题，放弃。
2. **逐页 `cudaMemcpyAsync`**：8192 次调用要 84–134 ms（约 15 µs/次的 API 开销），
   比 9.6 ms 的传输本身贵一个数量级。这条数据也反过来决定了 `MQA_GCHUNK` 的下限
   （单片不小于 1 MiB）。
3. **`cudaHostRegister` 调用方的 mmap 张量做零拷贝**：注册 256 MiB 要 13.8–15.3 ms，
   比它想省掉的 9.6 ms 传输还贵，而且每次调用张量地址都在换。
4. **gather 用 AVX2 非临时存储（`_mm256_stream_si256`）**：6.65–6.86 ms vs 6.81–6.97 ms，
   约 2%。说明 gather 瓶颈在随机页的读侧，不在写侧的 RFO。
5. **`OMP_WAIT_POLICY=passive`**：更慢（resident TC2 2955 → 3913 µs 方向恶化），
   master 线程发 DMA 时其他线程需要能立刻被唤醒。
6. **kernel 分块（`MQA_CHUNK`）**：想让 kernel 与 H2D 重叠，结果全面变慢 ——
   `MQA_CHUNK=64/128/256` 在 TC1 是 459/606/701，TC2 是 321/408/506，
   而整批一次启动是 920/768。把 kernel 插进拷贝流打断了 copy engine 的连续性，
   而它才是瓶颈；同时 `pages_used` 取 chunk 内最大值，切细以后空转 block 变多。
7. **更多流**：`MQA_STREAMS` 1/2/3/4 → TC2 775/771/769/769，基本持平。
   默认单 kernel 启动之后，多流已经没有什么可并行的了。
8. **更多 gather 线程**：8/16/32 线程下 TC2 的 gather 是 7.67/8.72/7.97 ms，8 最好。
   随机页 gather 早就打满了 DRAM，加线程只增加争用。

## 9. 复现

```bash
bash run.sh                      # 官方入口（sm_86，PTX JIT）
bash build_local.sh && ./main     # 本机原生 sm_120，避免 JIT
OMP_NUM_THREADS=8 ./main          # 报告里的配置
```

可选环境变量（全部有默认值，不设也能跑）：

| 变量 | 默认 | 作用 |
|---|---|---|
| `MQA_PROF` | 关 | 打印阶段耗时 |
| `MQA_STREAMS` | 2 | CUDA 流数量 |
| `MQA_GCHUNK` | `clamp(total_pages/24, 64, 512)` | gather/H2D 的页粒度 |
| `MQA_CHUNK` | 整批 | kernel 启动粒度 |
| `MQA_RESIDENT_KV` | 关 | 显存常驻页池（见第 7 节） |
| `MQA_FORCE_GENERIC` | 关 | 评测形状也走通用 kernel |

`OMP_NUM_THREADS=8` 是实测最优；再补 `OMP_PROC_BIND=close OMP_PLACES=cores`
能把 TC1 的抖动下限从 862 抬到 890 左右，但对中位数没有影响。

形状扫描（额外文件，不影响评测）：

```bash
nvcc -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 -Xcompiler -fopenmp -o shape_test shape_test.cu
OMP_NUM_THREADS=8 ./shape_test
```

## 10. 还没做的

- **让 kernel 与 H2D 真正重叠**：现在整批一次启动，kernel 必须等最后一片 KV 到齐。
  正确的做法应该是 kernel 在独立流上用 event 等每片 H2D，而不是把 kernel 塞回拷贝流
  （已验证后者更慢）。这是 TC1 剩余 40% 差距的主要来源。
- **`cp.async` 双缓冲 K 的 staging**：需要突破 48 KB 静态 shared memory 上限，
  改用动态 smem + `cudaFuncSetAttribute`。
- **v6 kernel 重排**：让一个 warp 负责 8 token × 全部 64 head，head 维归约留在寄存器里，
  可以去掉 `sPart` 和每页两次 barrier。主要利好 TC1 和常驻模式。

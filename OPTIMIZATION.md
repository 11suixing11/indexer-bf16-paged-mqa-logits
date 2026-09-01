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
| v6 | D2H 只回传活前缀 | ~920 | ~768 | 2.7e-14 / 5.9e-14 |
| v7 | 固定主机线程数（评测脚本不设 `OMP_NUM_THREADS`） | ~1000 | ~790 | 2.7e-14 / 5.9e-14 |
| **v8（最终）** | **输出直写 mapped pinned（取消 D2H）+ `cp.async` 页流水 + 页表预取** | **~1040** | **~798** | **2.675637e-14 / 5.917489e-14** |

最终相对 CPU 参考实现：**TC1 约 12.0×，TC2 约 9.0×**。
正确性余量：要求 `cos_diff < 5e-6`，实测 2.7e-14 / 5.9e-14，低了八个数量级；
输出 mask 逐元素校验通过。

单次调用耗时：TC1 约 1.03 ms，TC2 约 10.9 ms（含全部 H2D，因为 `main.cu`
的计时器包住了整个算子调用）。

多次运行的抖动（官方 `bash run.sh`，同一二进制多轮）：TC1 895–1136，
TC2 781–801；TC1 的中位数约 1000，落在 940–1060 之间的居多。TC1 数据量小、固定开销占比高，抖动主要来自主机侧
gather 和 CPU 频率，不是 kernel；TC2 贴着 PCIe 上限，非常稳。
下文所有对比数字都在同一台机器、同一组配置下取得。

## 2. 环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop (Blackwell, sm_120, 12 GiB) |
| CPU | 32 逻辑核 |
| 系统 | WSL2 Ubuntu 24.04，驱动 610.88，CUDA 13.3 |
| 编译 | `nvcc -O3 -std=c++17 -Xcompiler -fopenmp`，同时生成 sm_86 / sm_120 |
| 运行 | `bash run.sh`（无需任何环境变量；主机线程数由头文件自己定，见 v7） |

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

### v7：把主机线程数收回来（~940 → ~1000 / 790 GFLOPS）

这是全程单点收益最大、也最容易被忽略的一步：**`run.sh` 不设 `OMP_NUM_THREADS`**，
于是 OpenMP 默认开满 32 个线程，而我此前一直用 `OMP_NUM_THREADS=8 ./main` 调优。
两者差距极大 —— 同一个二进制，官方路径只有 TC1 638 / TC2 718，我自己跑却有
TC1 1025 / TC2 792。

| 线程数 | 4 | 6 | **8** | 10 | 12 | 16 | 32（默认） |
|---|---|---|---|---|---|---|---|
| TC1 | 991 | 1016 | **1032** | 1025 | 1009 | 955 | 658 |
| TC2 | 789 | 776 | **791** | 792 | 784 | 760 | 739 |

原因是 gather 受主机 DRAM 带宽限制，几个线程就打满了；再多的线程只会给每片
切片的 OpenMP barrier 增加汇合成本（一片 256 页时，32 线程每人只分到 8 页 = 128 KiB，
barrier 反而成了主要开销）。

所以在头文件里给三个并行域都加上 `num_threads(mqa::host_threads())`，默认取
`clamp(硬件线程数 / 4, 4, 16)`（本机得 8，换到内存通道更多的服务器会自动给更多），
可用 `MQA_THREADS` 覆盖。**评测者直接 `bash run.sh` 就能拿到调优后的成绩，
不依赖任何环境变量。**

### v8：输出直写 mapped pinned + `cp.async` 页流水 + 页表预取（~1000 → ~1040 / 798 GFLOPS）

三处改动：

1. **取消 D2H。** `h_out` 改用 `cudaHostAlloc(..., cudaHostAllocMapped)`，
   `cudaHostGetDevicePointer` 拿到设备侧别名直接交给 kernel 当 `logits` 指针，
   结果由 SM 经 PCIe 写回主机，那条 `cudaMemcpyAsync` 整个删掉。设备时间线里
   `d2h` 从 0.089 / 0.259 ms 直接变成 0.000（TC1 单次 +5.7%）。之所以值得删而不是
   优化：0.25 MiB 用了 0.089 ms，等效 2.8 GB/s，说明它几乎全是固定延迟。
   写回的可见性仍由 `cudaDeviceSynchronize` 保证，与原来靠拷贝时完全一样。
2. **`cp.async` 页流水。** K 的 global -> shared 从 LDG+STS 换成
   `cp.async.ca.shared.global`，并且把发起点挪到「上一页 mma 结束之后、epilogue
   之前」：那个 `__syncthreads` 本来就要用来发布 `sPart`，顺带也证明了所有 warp
   已经读完 `sK`，于是下一页的 16 KiB 可以立刻开始搬，与本页的归约、写回重叠。
   第 0 页则与 Q 的装载并行发起。**没有多用一个字节 shared memory**（双缓冲需要
   再来一份 17 KiB，会突破 48 KiB 静态上限并把 occupancy 砍半），每页的 barrier
   数量还从 3 个降到 2 个。TC1 kernel 0.160 -> 0.089 ms（−44%），TC2 0.45 -> 0.41 ms。
3. **页表预取。** 进页循环之前，一次性把本 block 负责的 4 个 `block_tables`
   表项读进寄存器（互相独立的 4 条 load，而不是每页一条依赖式 load），
   这样任何一页的数据搬运都不会再等自己的物理页号。

`cp.async` 的 `.ca` / `.cg` 两种缓存策略实测无差别（TC1 kern 中位数 0.116 vs 0.129，
TC2 0.408 vs 0.399，都在抖动范围内），保留 `.ca`。

## 5. 阶段耗时剖析

`MQA_PROF=1` 会在算子内部按阶段累计耗时（每个 testcase 独立计数，丢掉前 10 次
warm-up，因为那几次要付一次性的 `cudaMalloc`/`cudaHostAlloc`，`main.cu` 自己也把
第 0 次排除在平均之外）。最终版本（ms/call，90 次平均）：

| 阶段 | 含义 | TC1 | TC2 |
|---|---|---|---|
| plan | 扫 `block_tables` 建压缩计划、检查缓冲区容量 | 0.003 | 0.019 |
| meta | Q/weights/bt/cl 进 pinned 并发出上传 | 0.056 | 0.143 |
| gather | 主机把散页拷进 pinned 页池（与 DMA 重叠） | 0.245 | 7.521 |
| issue | 发起 kernel 与各次 copy 的 API 开销 | 0.403 | 0.459 |
| sync | `cudaDeviceSynchronize`，等剩余 DMA/kernel | 0.231 | 2.376 |
| out | 从 pinned 拷回 `output` 并写 `-inf` 尾巴 | 0.054 | 0.172 |
| **合计** | | **0.992** | **10.690** |

合计与 `main.cu` 测出的单次耗时基本吻合（阶段划分没有黑洞）。

**怎么读这张表**：TC2 的 `gather` 7.5 ms 看着吓人，但它是在 master 线程发出 DMA
之前记的，与 9.8 ms 的 H2D 完全重叠 —— 证据是 `gather + sync = 9.9 ms` 而不是
`7.5 + 9.8 = 17.3 ms`。主机侧的 `issue`/`gather` 都不在关键路径上，真正串行残留的
是最后一片 H2D 之后才能跑的 kernel。

### 设备时间线（`MQA_TIMELINE=1`）

上面那张表是主机视角，它只知道等了多久，不知道 copy engine 和 SM 分别在忙什么。
所以我又加了一套 event 时间线：在每次 H2D / kernel / D2H 前后各记一个 event
（event 对象池化复用，计时期间不再创建任何对象），按类型累加设备忙时，
再用最晚的 end event 减去起点得到整体跨度，差值就是设备空泡。90 次平均（ms）：

| | h2d | kern | d2h | busy | span | bubble |
|---|---|---|---|---|---|---|
| TC1 | 0.760 | 0.089 | 0.000 | 0.849 | 0.912 | 0.064（7%） |
| TC2 | 9.844 | 0.465 | 0.000 | 10.309 | 10.784 | 0.475（4%） |

这张表是后面所有判断的依据：**TC1 有 77% 的时间、TC2 有 91% 的时间，
设备上唯一在干的事情就是把 KV 从主机搬进来**。`d2h` 为 0 是 v8 直写 mapped
pinned 的结果。

它还顺手量出了 WSL2 的一项固定成本：单次 kernel 的 `kern` 区间约 0.09 ms，
而把同样的工作切成 2 / 4 / 8 次启动后变成 0.31 / 0.49 / 0.81 ms —— 每个 kernel
节点约 70 us、每个 copy 节点约 60 us 的下发延迟，都是实打实记在流上的。
这个数字直接判了「kernel 与 H2D 重叠」的死刑（见负结果）。

### 上界分析

| | TC1 | TC2 |
|---|---|---|
| 压缩后 KV + Q | 9 MiB | 130 MiB |
| H2D 理论下限 @14.2 GB/s | 0.665 ms | 9.60 ms |
| H2D 实测 | 0.760 ms | 9.844 ms |
| 实测总耗时 | 0.992 ms | 10.690 ms |
| 距 PCIe roofline | 67% | **90%** |

拆开看剩下的距离，TC1 的 0.992 ms = H2D 0.760 + kernel 0.089 + 设备空泡 0.064
+ 主机回写 0.054 + 零头；TC2 的 10.69 ms = 9.844 + 0.465 + 0.475 + 0.172 + 零头。
而这 0.089 / 0.465 的 kernel 区间里还各有约 0.07 ms 是 WSL2 的下发延迟，
真正的执行只有约 0.02 / 0.40 ms。也就是说：**即使把计算做到零，TC1 也只能再快
约 9%、TC2 约 4%**；这条实现已经贴在本平台的分页搬运上限上了。

唯一能突破的方向是让搬运本身和计算重叠，即把 kernel 切块、和 H2D 交错。
这条路我实测过两次，都是负的，而且时间线给出了确定的原因：切块之后
copy engine 的忙时完全不变（TC1 恒为 0.72 ms、TC2 恒为 9.8 ms），
变的是 kernel 与 D2H 的**设备**区间被下发延迟乘上了块数。
一个切口最多能藏起半个 kernel（TC1 约 0.045 ms），却要付约 0.13 ms 去开这个口子。
所以在 WSL2 上这条路是关着的；换到原生 Linux（下发延迟约 5 us）它会重新打开，
这一点我写在下面的负结果里，而不是假装它不存在。

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
| 默认（每次重传） | ~1040 GFLOPS | ~798 GFLOPS |
| `MQA_RESIDENT_KV=1` | **1219 GFLOPS** | **4966 GFLOPS** |

设备时间线（resident）：TC1 `h2d 0.270 | kern 0.149 | span 0.469`，
TC2 `h2d 0.206 | kern 0.419 | span 0.661`（TC1 的 h2d 是 Q 和 metadata，
它们每次调用都不一样，跳不掉）。TC2 的 4966 GFLOPS 就是这个 kernel 在
没有 PCIe 税时的真实水平；kern 0.419 ms 里还含约 0.07 ms 的 WSL2 下发延迟。
注意这个 kern 与默认模式下量到的 0.465 ms 基本一致 —— 两种模式跑的是同一个
kernel，只是数据来路不同，这也交叉验证了时间线的读数。

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
6. **kernel 分块（`MQA_CHUNK`）**：想让 kernel 与 H2D 重叠，结果全面变慢，
   而且加上时间线之后原因是确定的。`MQA_CHUNK=64/128/256/整批` 下：

   | | TC1 kern | TC1 d2h | TC1 GFLOPS | TC2 kern | TC2 d2h | TC2 GFLOPS |
   |---|---|---|---|---|---|---|
   | 整批（1 次启动） | 0.176 | 0.089 | 861 | 0.532 | 0.260 | 760 |
   | 256 | 0.306 | 0.200 | 714 | 4.475 | 2.539 | 485 |
   | 128 | 0.486 | 0.358 | 579 | 7.152 | 4.512 | 382 |
   | 64 | 0.812 | 0.682 | 431 | 10.572 | 7.427 | 293 |

   关键是 `h2d` 的忙时**一点没变**（TC1 恒 0.72 ms、TC2 恒 9.75 ms），
   膨胀的是 kernel 和 D2H 的设备区间：每个 kernel 节点约 70 us、每个 copy 节点
   约 60 us 的下发延迟，在 WSL2 上是记在流上的真实等待。
   我最初把原因写成"打断了 copy engine 的连续性"，时间线证明那是错的 ——
   copy engine 一直很顺，是启动本身太贵。一个切口最多藏起半个 kernel
   （TC1 约 0.045 ms），代价却是约 0.13 ms。原生 Linux 上下发延迟只有约 5 us，
   这条路会重新变得可行；这是本报告里唯一一个"平台决定结论"的地方。
   次要因素：`pages_used` 取 chunk 内最大值，切细以后空转 block 也变多。
7. **更多流**：`MQA_STREAMS` 1/2/3/4 → TC2 775/771/769/769，基本持平。
   默认单 kernel 启动之后，多流已经没有什么可并行的了。
8. **更多 gather 线程**：8/16/32 线程下 TC2 的 gather 是 7.67/8.72/7.97 ms，8 最好。
   随机页 gather 早就打满了 DRAM，加线程只增加争用。（线程数本身是个大坑，
   见第 4 节 v7 —— 官方脚本不设 `OMP_NUM_THREADS`，默认 32 线程会吃掉 TC1 三分之一。）
9. **切片大小渐进放大（`MQA_RAMP=1`）**：想法是第一片 gather 完全暴露在 DMA 之前，
   所以先来几片小的（64/128/256...）好让 copy engine 早点开工。实测 TC1 持平、
   TC2 反而差约 1%（768/774/776 → 765/768/760）。早期的小片排空得比 gather 补片
   还快，copy engine 在开头空转三次，亏得比省下的多。默认关闭。
10. **零拷贝直读主机 KV（把 gather 和 H2D 一起省掉）**：把调用方的 KV 竞技场
    `cudaHostRegister(..., cudaHostRegisterMapped)` 一次，让 kernel 直接按
    `block_tables` 从主机内存取页。微基准（512 MiB 竞技场、16 KiB 页）：
    8 MiB 工作集 11.4–12.5 GB/s，128 MiB 工作集 10.6–10.8 GB/s，
    顺序页与随机页几乎没差别；同一台机器上 pinned DMA 是 14.2–14.4 GB/s。
    对 TC1 大致打平（省掉 gather 与固定开销，但带宽差 13%），
    对 TC2 是 12.5 ms vs 9.8 ms，直接输 27%。SM 发起的 PCIe 读打不满链路，
    这条路只有在带宽差距被固定开销盖住时才划算，不作为主路径。
11. **`cp.async` 用 `.cg` 绕过 L1**：K 每页基本只读一次，理论上不该占 L1。
    实测与 `.ca` 在抖动范围内没有差别（TC1 kern 中位数 0.116 vs 0.129，
    TC2 0.408 vs 0.399），保留 `.ca`。
12. **打包 D2H（连续回传替代 `cudaMemcpy2DAsync`）**：以为 2D 拷贝的 0.096 ms
    是每行的建立开销，改成一次连续拷贝后还是 0.089 ms —— 它是延迟，不是带宽也不是
    per-row 开销。这一步本身没收益，但它给 v8 的"直写 mapped pinned"铺好了路
    （SM 侧的写就是连续的），所以保留。

## 9. 复现

```bash
bash run.sh                       # 官方入口（sm_86 + PTX JIT），不需要任何环境变量
bash build_local.sh && ./main     # 本机原生 sm_120，避免 JIT
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
| `MQA_TIMELINE` | 关 | 打印设备时间线（h2d / kern / d2h / 空泡） |
| `MQA_THREADS` | `clamp(hw/4, 4, 16)` | 主机并行域线程数 |
| `MQA_MAPOUT` | 开 | 输出直写 mapped pinned（关掉则退回 D2H 拷贝） |
| `MQA_RAMP` | 关 | 切片大小渐进放大（负结果，见第 8 节） |

主机线程数已经写在头文件里，不再依赖 `OMP_NUM_THREADS`；显式设置会被
`num_threads()` 覆盖，只有 `MQA_THREADS` 能改。补 `OMP_PROC_BIND=close
OMP_PLACES=cores` 能把 TC1 的抖动下限抬高一点，但对中位数没有影响。

形状扫描（额外文件，不影响评测）：

```bash
nvcc -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 -Xcompiler -fopenmp -o shape_test shape_test.cu
./shape_test
```

## 10. 还没做的

- **让 kernel 与 H2D 真正重叠**：现在整批一次启动，kernel 必须等最后一片 KV 到齐。
  在 WSL2 上这条路是关着的 —— 时间线量出每个 kernel 节点约 70 us 的下发延迟，
  开一个切口的代价（约 0.13 ms）大于它能藏起来的计算（约 0.045 ms），
  见第 8 节第 6 条。换到原生 Linux（下发延迟约 5 us）应该重新试一次：
  按现在的数字，TC1 大约还有 0.09 ms（9%）、TC2 大约 0.40 ms（4%）可拿。
- **`sK` 真正的双缓冲**：v8 只做到了"上一页算完就开始搬下一页"，
  单缓冲 + 一个 barrier。整页双缓冲需要再来一份 17 KiB，突破 48 KB 静态上限
  （要动态 smem + `cudaFuncSetAttribute`），而且会把 occupancy 从 2 block/SM
  砍到 1。正确的省内存写法是把一页切成两个 32 token 的半页缓冲，
  代价是 warp 分工要从"4 head 组 × 2 token 半区"改成"每 warp 8 token × 32 head"，
  epilogue 的跨 warp 归约也要跟着重写。收益上限只有几个百分点，先记下来。
- **kernel 重排去掉 `sPart`**：让一个 warp 负责 8 token × 全部 64 head，
  head 维归约完全留在寄存器里，可以去掉 `sPart` 和每页剩下的 2 次 barrier。
  与上一条冲突（8 warp × 8 token = 64 token 恰好是一整页，容不下半页缓冲），
  只能二选一。主要利好常驻模式，对评测的两个 case 影响在抖动量级。
- **主机侧 gather 与 DMA 的最后 4% 空泡**：TC2 还有 0.475 ms 的设备空泡，
  来自每片切片的 OpenMP barrier 与 master 线程发起 DMA 之间的间隙。
  用一个专职 issue 线程 + 每片原子计数器（而不是 barrier）应该能压掉一部分。

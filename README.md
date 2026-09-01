# indexer_bf16_paged_mqa_logits — CUDA 算子优化题目

## 项目简介

`indexer_bf16_paged_mqa_logits` 是大模型推理中 GQA/MQA 注意力计算 logits（注意力分数）的算子：
对分页（PagedAttention）存储的 KV Cache，为每个 (batch, next_n, head) 计算 query 与历史所有
KV token 的点积，经 ReLU + head 权重加权后在 head 维归约，得到每个 query 对应的 logits 序列。

本项目是 CUDA 算子优化基准工程。[indexer_mqa_logits.h](indexer_mqa_logits.h) 目前只调用 CPU
参考实现 `ref_bf16_paged_mqa_logits`，你的任务是在其中实现一个 CUDA kernel 并替换掉 ref 调用。

## 目录结构

| 文件 | 作用 | 能否修改 |
|------|------|---------|
| `indexer_mqa_logits.h` | 算子入口，你要实现 CUDA kernel 的地方 | ✅ 只能改这个 |
| `main.cu` | 主程序：生成测试数据、调用算子、正确性校验与性能计时 | ❌ |
| `ref_mqa_logits.h` | CPU 参考实现，用于对齐结果 | ❌ |
| `testcase.h` | 测试用例参数与数据生成 | ❌ |
| `Tensor.h` | 多维张量数据结构 | ❌ |
| `allocator.h` | 主机端内存分配器 | ❌ |
| `utils.h` | bf16 类型与基础工具函数 | ❌ |
| `run.sh` | 编译并运行 | ❌ |

## 算法说明

### 输入输出

- `q`          ：`[batch_size, next_n, num_heads, dim]`，bf16
- `kv_cache`   ：`[num_blocks, block_size, 1, dim]`，bf16，按 block 分页存储
- `block_tables`：`[batch_size, max_num_blocks]`，每行的 KV 页物理块号映射
- `context_lens`：`[batch_size]`，每个 batch 当前的上下文长度
- `weights`    ：`[batch_size * next_n, num_heads]`，head 权重
- `output`     ：`[batch_size * next_n, max_model_len]`，输出 logits（fp32）

### 计算流程

对每个 `(batch_idx, n)`，令 `bn = batch_idx * next_n + n`、`context_len = context_lens[batch_idx]`：

1. 通过 `block_tables` 把分页的 `kv_cache` 按逻辑顺序 gather 成连续的 K；
2. 对每个 head `h`，计算 query `q[bn, h, :]` 与每个 token 位置 `t` 的 K 行点积得到分数
   `s[t] = max(0, q[bn,h,:] · K[t,:]) * weights[bn, h]`；
3. 在 head 维累加，得到 `output[bn, t] = Σ_h s[t]`；
4. 越界位置（`t >= context_len` 或 `t > context_len - next_n + n`）置为 `-INFINITY`。

### 评测

- 正确性：`cos_diff < 5e-6`，且 mask（`-INFINITY` / `NaN`）与参考一致；
- 性能：`main.cu` 统计多轮平均耗时并打印 GFLOPS；
  - TC1: `batch=32, next_n=2, avg_kv=1024`
  - TC2: `batch=128, next_n=1, avg_kv=4096`
- 参数：`block_size=64, num_heads=64, dim=128, max_model_len=8192`。

## 运行

```bash
bash run.sh
```

等价于手动编译运行（RTX 3050 为 sm_86，其他 GPU 请替换 `-arch=sm_xx`）：

```bash
nvcc -O3 -arch=sm_86 -std=c++17 -Xcompiler -fopenmp -o main main.cu
./main
```

程序依次运行两个测试用例，输出 `cos_diff`（正确性）与平均吞吐（GFLOPS）。

## 优化方向提示

- 把 K gather 结果缓存到 Shared Memory，避免每个 head 重复读全局内存；
- 分块计算 Q·K^T，配合 Tensor Core（`mma.sync` / WMMA）做 BF16 矩阵乘；
- 处理 Bank 冲突（Shared Memory Padding / swizzle）；
- 计算与访存流水线重叠、多 batch 并行调度。

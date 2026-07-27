# MUSA Custom All Gather 使用指南

## 适用范围

该路径使用 MUSA IPC memory handle 和机内 JIT kernel，因此只适用于同一台机器内的
通信 group。跨机 group 不会启用 Custom All Gather，而是回退到原有 MCCL 或
`torch.distributed` 路径。

当前支持的 Custom All Gather group size 为 `2`、`4`、`6`、`8`。以下情况会自动回退：

- 输入为空或输入输出不在同一张 MUSA 卡上
- 输入输出 dtype 不一致
- 输入或输出不是满足要求的连续布局
- 输入字节数不是 16 字节对齐
- 输入或输出地址不是 16 字节对齐
- 输入字节数超过 `SGLANG_CUSTOM_AG_MAX_SIZE_BYTES`
- group 不是 node-local group

回退是按调用发生的，因此同一个服务中可能同时出现 Custom All Gather 和 MCCL。

## 启用和关闭

启用时，在启动每个 rank 进程前设置：

```bash
export SGLANG_MUSA_USE_JIT_ALL_GATHER=1
export SGLANG_CUSTOM_AG_MAX_SIZE_BYTES=83886080
export SGLANG_CUSTOM_AG_THREADS=512
export SGLANG_CUSTOM_AG_BLOCKS=48
```

关闭时使用：

```bash
export SGLANG_MUSA_USE_JIT_ALL_GATHER=0
```

不应使用旧的 `SGLANG_CP_USE_CUSTOM_ALL_GATHER` 或
`SGLANG_CP_CUSTOM_ALL_GATHER_MAX_SIZE`；当前源码不读取这两个变量。

`SGLANG_CUSTOM_AG_MAX_SIZE_BYTES=83886080` 是当前 serving 测试采用的 80 MiB 输入
阈值。超过该阈值的调用会回退到 MCCL，不代表 Custom All Gather 支持任意大消息。

## 启动示例

在已安装当前分支的 MUSA 环境中：

```bash
source /root/.virtualenvs/sglang-default/bin/activate
export PYTHONPATH=/path/to/sglang/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/usr/local/musa-4.3.5/lib:/usr/lib/x86_64-linux-gnu
export SGLANG_MUSA_USE_JIT_ALL_GATHER=1
export SGLANG_CUSTOM_AG_MAX_SIZE_BYTES=83886080
export SGLANG_CUSTOM_AG_THREADS=512
export SGLANG_CUSTOM_AG_BLOCKS=48

python3 -m sglang.launch_server \
  --model-path /path/to/model \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 24586
```

双机启动时，仍然需要使用正常的分布式参数。Custom All Gather 不需要、也不能替代
跨机初始化或跨机 PP 通信。

建议打开 debug 日志确认路径：

- node-local group 应出现 Custom All Gather 初始化日志
- 非 node-local group 应出现 skip/fallback 日志
- 初始化失败时服务会保留 MCCL 回退路径，并提示设置
  `SGLANG_MUSA_USE_JIT_ALL_GATHER=0`

## 内核 benchmark

仓库内的独立 benchmark 会用 Torch/MCCL All Gather 作为对照，并检查结果正确性：

```bash
PYTHONPATH=$PWD/python \
LD_LIBRARY_PATH=/usr/local/musa-4.3.5/lib:/usr/lib/x86_64-linux-gnu \
LOCAL_WORLD_SIZE=8 \
GPUS_PER_NODE=8 \
SGLANG_MUSA_USE_JIT_ALL_GATHER=1 \
MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python benchmark/musa/bench_custom_allgather_graph.py \
  --world-size 8 \
  --hidden 2048 \
  --dtype bf16 \
  --impls torch jit \
  --modes eager \
  --tokens 1 8 128 1024 8192 16384 \
  --warmup 10 \
  --iters 50 \
  --repeats 3 \
  --isolate-impls \
  --output-csv /tmp/custom_all_gather_vs_mccl.csv \
  --output-overwrite
```

`torch_ag_eager` 是 MCCL 对照，`jit_cag_eager` 是当前 serving 使用的 eager
Custom All Gather 路径。`registered` 和 `graph` 是额外实验路径，不能直接代表关闭
CUDA Graph 的普通 serving 性能。

如果需要测试超过当前 serving 阈值的消息，显式传入更大的 `--max-size`，并在结果中
记录该配置：

```bash
--max-size 536870912
```

## 正确性检查

MUSA 单元测试位于：

```text
test/srt/musa/test_musa_custom_all_gather.py
```

测试覆盖多种 dtype、eager、registered、MUSA Graph、fallback 和较大输入。运行方式：

```bash
pytest -q test/srt/musa/test_musa_custom_all_gather.py
```

没有 `torch_musa` 或 MUSA 设备时测试会被跳过；这不等价于通过 GPU 正确性验证。

## 本次同机 benchmark 结果

本节记录基于最新 `musa/0.5.12.post1` 加 Custom All Gather 提交的单机八卡对比。
测试环境：

- 机器：`10.20.34.83`，8 卡 MTT S5000，驱动 `3.3.5-server`
- 镜像：`registry.mthreads.com/mcconline/inference/sglang:v0.5.12.post1-ph1-4.3.5-torch2.9.0-20260710`
- Python：`/root/.virtualenvs/sglang-default/bin/python3`
- 分支提交：`460e6e6bf`，基于 `origin/musa/0.5.12.post1@13c5cd049`
- `world_size=8`、`hidden=2048`、`dtype=bf16`
- `SGLANG_CUSTOM_AG_MAX_SIZE_BYTES=83886080`、`threads=512`、`blocks=48`
- `warmup=10`、`iters=50`、`repeats=3`，MCCL 和 Custom AG 在独立 worker group 中分别运行
- 对照：`torch_ag_eager` 使用 MCCL，`jit_cag_eager` 使用 Custom AG eager

正值表示 Custom AG 相对 MCCL 的改善；延迟下降和带宽上升均记为正值。

| tokens | bytes | MCCL latency (ms) | Custom AG latency (ms) | 延迟改善 | MCCL bus BW (GB/s) | Custom AG bus BW (GB/s) | 带宽改善 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 KiB | 0.0983 | 0.0221 | +77.48% | 0.04 | 0.16 | +343.96% |
| 8 | 32 KiB | 0.1485 | 0.0284 | +80.86% | 0.19 | 1.01 | +422.60% |
| 128 | 512 KiB | 0.1923 | 0.0650 | +66.19% | 2.39 | 7.06 | +195.79% |
| 1024 | 4 MiB | 0.2917 | 0.1453 | +50.18% | 12.58 | 25.25 | +100.72% |
| 8192 | 32 MiB | 1.2388 | 1.0497 | +15.26% | 23.70 | 27.97 | +18.01% |
| 16384 | 64 MiB | 2.2650 | 2.1024 | +7.18% | 25.93 | 27.93 | +7.73% |
| 32768* | 128 MiB | 4.3430 | 4.0990 | +5.62% | 27.04 | 28.65 | +5.95% |

本轮所有 case 的 tensor correctness check 通过，benchmark 正常退出。由于生产阈值为
80 MiB，前六个 case 均走生产配置；128 MiB case 使用单独的
`--max-size 536870912` 补测，不能直接作为 80 MiB serving 阈值下的结果。

128 MiB 补测命令只需将生产 benchmark 命令替换为：

```bash
--tokens 32768 \
--max-size 536870912 \
--output-csv /tmp/custom_all_gather_128m.csv
```

结论：在单机八卡、node-local、eager 路径下，Custom AG 的 kernel 通信延迟稳定低于
MCCL。消息越大，绝对收益仍存在但相对收益逐渐收敛；64 MiB 时延迟改善为 `7.18%`，
128 MiB 在放宽阈值后仍改善 `5.62%`。
该结果证明通信内核有收益，但不等价于 serving 端到端一定有同等收益，因为 serving
还包含调度、attention、DeepEP、同步和可能的 fallback。

原始文件：

```text
/home/dist/jzxue/remote-workdir/custom_all_gather_results/20260724_musa_0.5.12.post1_custom_ag_vs_mccl/production_threshold.csv
/home/dist/jzxue/remote-workdir/custom_all_gather_results/20260724_musa_0.5.12.post1_custom_ag_vs_mccl/production_threshold.jsonl
```

SHA256：

```text
c6af66213679c5fd1ec2660811364e7e852c06e4b1e1eb29f6ed22f510bd3424  production_threshold.csv
87460edcccf7dd8b10a423068bcfd3de47c2f6c8f0b93d47aafb5a87a16cab7d  production_threshold.jsonl
791903490be375d42d9289de23d1c2ffe31f1086c62ef491d1305014227f0fed  large_128m.csv
7696cca429d7154bc97d4cc35b0ffc8779008fa22728b51a228b273fb5ee3db5  large_128m.jsonl
```

benchmark 原始 CSV、运行日志和租约信息应与本节一同保存，避免只保留汇总比例。

## 排障

1. 确认所有 rank 使用同一份源码和同一 Python 环境。
2. 确认所有 rank 的 `SGLANG_MUSA_USE_JIT_ALL_GATHER` 值一致。
3. 确认跨机 group 没有被误判为 node-local；检查 `LOCAL_WORLD_SIZE` 或
   `GPUS_PER_NODE` 是否正确。
4. 对比输入字节数和 `SGLANG_CUSTOM_AG_MAX_SIZE_BYTES`，确认没有发生静默回退。
5. 先用独立 benchmark 验证 Custom AG 与 MCCL 的结果一致，再分析 serving 的 TTFT、
   E2E 和 DeepEP 成本。
6. 性能 benchmark 必须使用 GPU lease；完成后释放租约，避免空占资源。

## DeepSeek-V2-Lite 端到端验证

- [x] 在单机 8 卡上使用宿主机模型目录 `/ipfs/models/DeepSeek-V2-Lite`（容器内为
  `/home/dist/models/DeepSeek-V2-Lite`），固定模型、TP/EP、attention backend、
  chunked prefill 和请求集，分别运行 MCCL 与 Custom AG 服务。
- [x] 对比两组服务的 TTFT、input token/s 和 E2E latency，并从服务日志确认 Custom AG
  JIT 模块已编译、加载。
- [x] 保存完整启动参数、环境变量、服务日志、benchmark JSONL 和 GPU lease 信息；
  确认收益来自 Custom AG，而不是机器组合、warmup、缓存或请求顺序差异。

### Dry-run

仓库提供了一个默认不占用 GPU 的 A/B 脚本：

```bash
DRY_RUN=1 bash benchmark/musa/run_custom_allgather_e2e_ab.sh
```

dry-run 会打印 MCCL 和 Custom AG 两套服务启动命令、ready check、warmup 和以下固定
请求，不会创建结果目录或启动服务：

随机请求默认使用固定数据集
`/home/dist/jzxue/datasets/ShareGPT_V3_unfiltered_cleaned_split.json`，避免 benchmark
运行时访问 Hugging Face。

| case | input | output | prompts | concurrency |
|---|---:|---:|---:|---:|
| `prefill_8k_1` | 8192 | 1 | 4 | 1 |
| `mixed_4k_32` | 4096 | 32 | 8 | 4 |
| `decode_1k_128` | 1024 | 128 | 4 | 1 |

两组服务固定使用 `TP=8`、`EP=1`、FA3、chunked prefill 8192、
`MUSA_LAUNCH_BLOCKING=1`，并关闭 CUDA Graph、overlap schedule、radix cache 和
custom all reduce。该测试只验证单机 tensor-parallel all-gather，不启用 DeepEP，避免
expert-parallel 通信成为额外变量。脚本统一设置
`SGLANG_MUSA_MOE_GEMV_SWIGLU_MAX_TOKENS=0`，避免短 token batch 进入当前不支持
DeepSeek-V2-Lite 权重形状的 MUSA MoE BF16 GEMV。两组之间只切换
`SGLANG_MUSA_USE_JIT_ALL_GATHER`。

实际执行前必须先获得单机 8 卡独占租约，然后显式传入租约 ID：

```bash
GPU_LEASE_ID=<lease-id> \
DRY_RUN=0 \
RESULTS_DIR=/home/dist/jzxue/remote-workdir/custom_all_gather_results/e2e-deepseek-v2-lite \
bash benchmark/musa/run_custom_allgather_e2e_ab.sh
```

边界测试可以通过环境变量选择请求集、重复次数和 mode 顺序：

```bash
GPU_LEASE_ID=<lease-id> \
DRY_RUN=0 \
CASE_PROFILE=prefill_final \
BENCH_REPEATS=3 \
MODE_ORDER=mccl,custom_ag \
RESULTS_DIR=/home/dist/jzxue/remote-workdir/custom_all_gather_results/e2e-boundary \
bash benchmark/musa/run_custom_allgather_e2e_ab.sh
```

`CASE_PROFILE` 支持 `default`、`boundary`、`stability`、`prefill_boundary`、
`prefill_long` 和 `prefill_final`。`MODE_ORDER` 支持 `mccl,custom_ag` 与
`custom_ag,mccl`，用于排除服务启动顺序影响。当 `BENCH_REPEATS` 大于 1 时，结果文件
增加 `_rN` 后缀，避免后一次覆盖前一次。

脚本在 `DRY_RUN=0` 时会检查租约 ID、模型目录、Python 和 curl；任一条件不满足会在
启动服务前退出。租约有效性仍需通过 `gpu-lease` MCP 或看板确认，脚本不会绕过租约
系统自行占卡。

### 首轮功能验证

2026-07-24 在同一台单机八卡环境完成 A/B。正值表示 Custom AG 相对 MCCL 改善；
TTFT 和 E2E 按延迟下降计算，input tok/s 按吞吐上升计算。

| case | MCCL TTFT (ms) | Custom AG TTFT (ms) | TTFT 改善 | MCCL input tok/s | Custom AG input tok/s | 吞吐改善 | MCCL E2E (ms) | Custom AG E2E (ms) | E2E 改善 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `prefill_8k_1` | 375.38 | 370.32 | +1.35% | 21627.70 | 22012.43 | +1.78% | 375.39 | 370.33 | +1.35% |
| `mixed_4k_32` | 576.49 | 441.72 | +23.38% | 3425.95 | 3533.70 | +3.15% | 4777.96 | 4631.92 | +3.06% |
| `decode_1k_128` | 146.15 | 148.78 | -1.80% | 67.93 | 67.76 | -0.24% | 15072.07 | 15108.61 | -0.24% |

本轮 Custom AG 对 prefill 和 mixed case 有正向收益，长 decode 的吞吐和 E2E 差异在
0.25% 以内。每个 mode 只运行一次，表格用于功能和初步性能验证，不作为稳定性能结论。
`custom_ag_server.log` 中可以看到
`sglang_musa_custom_all_gather_t512_b48_mb120_ab1_db0` 的编译和加载记录；当前服务日志
没有逐 group 的命中、消息大小和 fallback 计数，因此本轮不能据此统计这些次数。

原始结果位于：

```text
/home/dist/jzxue/remote-workdir/custom_all_gather_results/20260724_deepseek_v2_lite_e2e_3480_retry11
```

### 稳定性边界复测

为排除首轮单次结果的偶然性，随后在同一台单机八卡环境完成以下测试：

1. 全量扫描 prefill 长度以及 mixed/decode concurrency 1、2、4、8，并反转
   MCCL/Custom AG 启动顺序。
2. 对候选 case 增加五轮重复。
3. 对 prefill 64、128、256、512、768、1024 和 16K 使用长窗口 AB/BA。
4. 最后固定每个 case 256 个请求，对 128、256、768 和 1024 再运行三轮。

下表正值表示 Custom AG 的 E2E latency 更低。长窗口的预设门槛为至少 4/5 轮正向且
中位数为正；严格稳定则要求 5/5 正向。

| case | 观测数 | 正向轮数 | E2E 改善中位数 | 最差 | 最好 | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `prefill_64_1` | 2 | 0 | -0.66% | -0.68% | -0.65% | 负向 |
| `prefill_128_1` | 5 | 4 | +1.35% | -2.57% | +3.44% | 唯一达到 4/5 门槛 |
| `prefill_256_1` | 5 | 3 | +0.30% | -0.94% | +1.47% | 不稳定 |
| `prefill_768_1` | 5 | 3 | +0.28% | -1.11% | +2.55% | 不稳定 |
| `prefill_1k_1` | 5 | 2 | -0.24% | -1.93% | +3.03% | 无稳定收益 |
| `mixed_4k_32_c4` | 8 | 6 | +2.09% | -1.69% | +6.20% | 峰值高但不稳定 |
| `decode_1k_128_c8` | 8 | 5 | +0.22% | -1.22% | +5.42% | 接近持平 |

当前能够确认的端到端正收益边界是：

```text
模型: DeepSeek-V2-Lite
并行: TP=8, EP=1, 单机
workload: prefill only
input/output: 128 / 1
concurrency: 1
长窗口请求数: 256
chunked prefill: 8192
DeepEP: off
```

在该配置下，Custom AG 的 E2E 改善中位数为 `+1.35%`，input throughput 改善中位数
为 `+1.36%`，五轮中的最大 E2E 改善为 `+3.44%`。这是本轮满足 4/5 门槛的最大稳定
收益；若严格要求每轮都正向，则本轮没有任何配置达标。因此当前结果只支持在明确的
128-token prefill workload 上按 A/B 验证后开启，不支持 serving 全局默认开启。

64 token 与 256 token 均未形成连续正向区间，说明边界更可能由实际 AllGather shape、
对齐和 fallback 共同决定，而不是单纯由请求长度决定。当前服务日志没有逐调用 hit、
bytes 和 fallback 计数，后续若要把 128-token workload 映射为通用通信边界，需要先补充
这些统计。

本轮原始结果目录：

```text
/home/dist/jzxue/remote-workdir/custom_all_gather_results/20260724_deepseek_v2_lite_e2e_3571_boundary_ab1
/home/dist/jzxue/remote-workdir/custom_all_gather_results/20260724_deepseek_v2_lite_e2e_3571_boundary_ba23
/home/dist/jzxue/remote-workdir/custom_all_gather_results/20260724_deepseek_v2_lite_e2e_3571_stability5
/home/dist/jzxue/remote-workdir/custom_all_gather_results/20260724_deepseek_v2_lite_e2e_3571_prefill_boundary5
/home/dist/jzxue/remote-workdir/custom_all_gather_results/20260724_deepseek_v2_lite_e2e_3571_prefill_long_ab
/home/dist/jzxue/remote-workdir/custom_all_gather_results/20260724_deepseek_v2_lite_e2e_3571_prefill_long_ba
/home/dist/jzxue/remote-workdir/custom_all_gather_results/20260724_deepseek_v2_lite_e2e_3571_prefill_final3
```

# Common Tools

These are the tools commonly used while optimizing MUSA kernels in SGLang.

## Environment And Device

| Tool | Use |
|---|---|
| `MUSA_VISIBLE_DEVICES=<id>` | Pin benchmark or test to one MUSA device |
| `python -c "import torch; print(torch.musa.get_device_capability())"` | Check MUSA arch/capability from Python |
| `mthreads-gmi` or `mthreads-gmi -q` | Inspect device utilization, memory use and process ownership when available |
| `musaInfo` | Inspect MUSA device/runtime information when available |
| `musa_version_query` | Check installed MUSA stack versions when available |

## Code Search And Repo Inspection

| Tool | Use |
|---|---|
| `rg "pattern" path` | Fast source search |
| `rg --files path` | Fast file listing |
| `sed -n 'start,endp' file` | Read focused file sections |
| `find path -maxdepth N -type f` | Inspect directory structure |
| `git diff -- file` | Review local edits without touching unrelated work |
| `git status --short` | Check dirty worktree state |

## Correctness And Smoke Tests

| Tool | Use |
|---|---|
| focused Python reproducer | Validate one kernel and one shape before broad testing |
| PyTorch reference implementation | Source of truth for numerical correctness |
| existing SGLang wrapper | Verify API-level mutation and dispatch semantics |
| `torch.musa.synchronize()` | Force async errors to surface at the intended boundary |
| tiny deterministic tensors | Debug layout, masks, tails and reductions |
| full target-shape tensors | Confirm the optimized path remains correct at scale |

## Benchmarking

| Tool | Use |
|---|---|
| `mate.testing.utils.bench_kineto` | Preferred kernel timing for MUSA csrc/JIT paths and named-kernel capture |
| `flush_l2=True` in `bench_kineto` | Cold-cache bandwidth measurements |
| `with_multiple_kernels=True` in `bench_kineto` | Correct timing for wrappers that launch multiple kernels |
| TileLang `get_profiler()` | TileLang kernel correctness and latency measurement |
| TileLang `do_bench()` | TileLang median latency measurement |
| `sglang/benchmark/musa/*.py` | Local benchmark harnesses for SGLang MUSA kernels |
| Markdown output tables | Preferred benchmark artifact for comparing shapes/providers |

## Profiling And Trace

| Tool | Use |
|---|---|
| MATE kineto output | Identify kernel names, latency and timeline behavior |
| SGLang server profile trace | End-to-end prefill/decode trace validation |
| Perfetto UI | Inspect Chrome trace files |
| `chrome://tracing` | Alternative Chrome trace viewer |
| Moore Perf / Roofline tools | Top-down bottleneck and utilization analysis when available |

## Generated Code And Assembly

| Tool | Use |
|---|---|
| TileLang dump environment variables | Dump generated MUSA source for TileLang kernels |
| `mcc -S` | Produce compiler assembly when direct MusaAsm tooling is unavailable |
| `musaasm` | Inspect MusaAsm output when available |
| compiler command logs | Confirm arch flags, include paths and optimization flags |
| generated `.mu` / MUSA source | Verify vectorization, loop shape, masks and barriers before assembly-level tuning |

## Runtime Debugging

| Tool | Use |
|---|---|
| `musaGetLastError` checks in csrc | Catch launch failures close to the kernel call |
| TVM FFI checks | Validate dtype, device, contiguity and shape at wrapper boundary |
| SGLang kernel API logging | Capture kernel call inputs before crash when available |
| `CUDA_LAUNCH_BLOCKING` equivalent is not assumed | Prefer explicit MUSA synchronization and local reproducer |
| small-shape binary search | Locate first failing shape or dispatch branch |

## Typical Commands

```bash
source /root/.virtualenvs/sglang-0.5.6/bin/activate
export MUSA_VISIBLE_DEVICES=6
```

Run a local benchmark script:

```bash
cd /sgl-workspace/sglang/benchmark/musa
python bench_jit_kernels.py --num-tests 5
```

Search for a dispatch branch:

```bash
rg -n "dispatch|__global__|bench_kineto|musaGetLastError" \
  /sgl-workspace/sglang/python/sglang/srt/hardware_backend/musa
```

Inspect local edits:

```bash
git status --short
git diff -- python/sglang/srt/hardware_backend/musa
```

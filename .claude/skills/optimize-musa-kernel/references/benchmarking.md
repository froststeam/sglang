# Benchmarking And Reporting

## Measurement Rules

| Rule | Why |
|---|---|
| Validate correctness before profiling | Fast wrong kernels are not useful |
| Measure the real dispatched path | Avoids benchmarking a path that production never calls |
| Keep synchronization policy identical | Baseline and candidate must include the same async boundaries |
| Use cold-cache timing for bandwidth claims | Warm-cache data can hide real serving/prefill traffic |
| Name exact kernel symbols when using profiler tools | Confirms the intended kernel was captured |
| Use multi-kernel measurement for fused wrappers that launch multiple kernels | Prevents partial timing |

## Required Metadata

Always report:

- kernel path and dispatch branch
- comparator
- shape, dtype, layout, strides if relevant
- device id and relevant environment variables
- warmup, reps, timing backend, synchronization policy
- correctness tolerance and max diff if applicable
- bandwidth or TFLOPS formula

## Bandwidth Formula Rules

Use conservative logical bytes:

- Count input reads and output writes required by the operator contract.
- Count scale, metadata, residual, state, and auxiliary tensors only when they
  are actually read or written per measured invocation.
- Do not count reused weights or state as per-token traffic unless the kernel
  reloads them per token/chunk.
- Separate lower-bound logical bandwidth from profiler-reported bandwidth.
- Treat values above plausible hardware limits as a formula bug until proven
  otherwise.

## Report Template

```text
Kernel path:
Dispatch branch:
Comparator:
Shape/dtype/layout:
Timing:
Correctness:
Bandwidth formula:

| Config | Correct | Latency | BW/TFLOPS | Notes |
|---|---|---:|---:|---|
| baseline | yes | ... | ... | ... |
| candidate | yes/no | ... | ... | ... |
```

---
name: optimize-musa-kernel
description: General workflow for writing, validating, profiling, assembly-auditing, and optimizing MUSA kernels in SGLang. Use for MUSA/Torch-MUSA/TileLang/csrc kernels when improving latency, bandwidth, occupancy, vectorized load/store, launch shape, reductions, cache behavior, compiler flags, or CUDA/Triton-to-MUSA ports. This skill is intentionally kernel-agnostic.
---

# Optimize MUSA Kernel

Use this skill for MUSA kernel work that needs a disciplined end-to-end loop:
define the operator contract, make the smallest correct implementation, measure
the real dispatched path, inspect generated code or assembly when needed, and
optimize one hypothesis at a time.

## Workflow

1. Triage the task.
   Decide whether the work is correctness, benchmarking, codegen/assembly,
   compiler flags, porting, or integration. Load only the matching references
   from the Read Path.

2. Define the exact contract.
   Record shape, dtype, layout, strides, mutation behavior, optional arguments,
   output conventions, invalid-index behavior, and graph-capture constraints.

3. Identify the source of truth.
   Compare against PyTorch, an existing SGLang path, Triton, FlashInfer, csrc,
   TileLang, or a previous MUSA implementation. Do not optimize against a
   simplified reference unless the task explicitly asks for that.

4. Implement the minimal correct path.
   Start with a simple scalar, row, warp, subwarp, or CTA mapping. Add tiling,
   vectorization, fusion, cache hints, or inline assembly only after correctness
   passes.

5. Validate correctness before profiling.
   Test tiny deterministic shapes first, then target shapes. Check optional
   arguments, masks, boundary/tail handling, auxiliary outputs, and dtype/layout
   variants covered by dispatch.

6. Benchmark the actual dispatched kernel.
   Record environment, device, shape, dtype, layout, warmup/reps, timing backend,
   synchronization policy, kernel names, comparator, and bandwidth or TFLOPS
   formula.

7. Inspect generated code or assembly for low-level claims.
   Audit LSU width, vectorization, cache hints, address math, spills, barriers,
   warp/subwarp reductions, and expensive math instructions.

8. Optimize one hypothesis at a time.
   Change exactly one launch shape, thread count, vector width, tile size,
   branch split, cache hint, fusion decision, or compile flag. Re-run
   correctness and benchmark the target shape.

9. Integrate conservatively.
   Keep fast-path guards explicit, preserve public API semantics, avoid
   graph-unsafe host sync, and keep feature-complete fallbacks correct.

## Read Path

Open only what is needed for the current task:

- Common tools: `references/tools.md`
- Correctness failures or crashes: `references/debugging-correctness.md`
- Benchmarking and reporting: `references/benchmarking.md`
- Assembly and generated-code audit: `references/assembly-audit.md`
- Compiler flag experiments: `references/compile-options.md`
- Optimization levers: `references/optimization-levers.md`
- CUDA/Triton to MUSA porting notes: `references/cuda-to-musa.md`
- Submission checklist: `references/submission-checklist.md`

## Hot-Path Checklist

| Area | Check |
|---|---|
| Launch shape | Expose independent axes; avoid serializing obvious row/head/group dimensions inside one CTA |
| Thread mapping | Know whether each thread owns one scalar, vector lane, channel group, token, row, or tile |
| Memory width | Verify vectorized paths emit wide loads/stores, not scalar LSU |
| Reductions | Prefer warp/subwarp reductions for small independent groups; use CTA reductions only when needed |
| Tail handling | Keep boundary checks correct without taxing the common aligned path |
| Cache behavior | Measure before changing cache hints; use cold-cache timing for bandwidth claims |
| Address math | Remove repeated divisions, modulos, and 64-bit address work from hot loops when possible |
| Register pressure | Watch spills when adding vector lanes, unrolling, or local arrays |
| Expensive math | Treat `exp`, `tanh`, reciprocal refinement, softmax, sigmoid, and SiLU as instruction-bound work |
| Graph safety | Do not add `.item()`, `.cpu()`, broad host sync, or torch fallback in graph-captured runtime paths |

## Result Template

```text
Kernel path:
Dispatch branch:
Comparator:
Shape/dtype/layout:
Timing backend:
Warmup/reps:
Synchronization:
Correctness:
Bandwidth or TFLOPS formula:
Compile flags:
Assembly/codegen observation:

| Config | Correct | Latency | BW/TFLOPS | Notes |
|---|---|---:|---:|---|
| baseline | yes | ... | ... | ... |
| candidate | yes/no | ... | ... | ... |
```

## Done Criteria

- Correctness is validated on the intended MUSA environment.
- The measured path is the real dispatched path.
- Benchmark metadata and formulas are recorded.
- Assembly/generated code is inspected when performance claims depend on it.
- Rejected non-obvious optimization attempts are recorded.
- Fast-path assumptions are explicit in dispatch guards.
- Runtime path remains graph-safe.

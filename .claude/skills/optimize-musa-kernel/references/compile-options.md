# MUSA Compile Options

This note summarizes MUSA kernel compile options and explains what each option
is for, which generic kernel patterns it usually matches, and the main risk.
Treat backend-specific `-mllvm` options as compiler-version-sensitive: keep only
measured wins on the target MCC/MUSA stack.

## Typical JIT Build Shape

A normal JIT path usually builds MUSA sources with a command shape like:

```bash
mcc -MD -MF $out.d $cuda_cflags -c $in -o $out
```

The common `$cuda_cflags` are:

```text
-fPIC
-std=c++17
-O2
--offload-arch=mp_${ARCH}
-I...
```

Then each kernel may append its own device flags, and users may append more
flags through the build system's environment hook when available.

## Common Options

| Option | Function | Kernel scenario | Risk or note |
|---|---|---|---|
| `-fPIC` | Generate position-independent code. | Required when objects are linked into Python/JIT `.so` libraries. | Usually keep enabled for extension modules. |
| `-std=c++17` | Compile as C++17. | Modern C++ host/device code and template-heavy kernel libraries. | Use the same standard across linked objects. |
| `-O2` | General optimization level. | Good baseline for most extension kernels: host wrappers, dispatch glue, moderate device code, and kernels where compile time and register pressure should stay controlled. | Often combined with device-specific `-Od3`; do not assume last optimization flag semantics without checking MCC behavior. |
| `-O3` | More aggressive optimization level. | Small or regular kernels with simple loops, row-wise reductions, elementwise math, scans, or generated code where stronger unrolling and scalar simplification may help. | Can increase compile time, register pressure, and code size; may hurt occupancy. |
| `-Od3` | MUSA device-side optimization level used by many performance kernels. | Handwritten high-performance `.mu` kernels with carefully structured hot loops, tiled loads, and repeated arithmetic. | MCC-specific; validate on the deployed compiler. |
| `--offload-arch=mp_${ARCH}` | Target a specific MTGPU architecture, for example `mp_31`. | All architecture-specific MUSA builds. | Wrong arch can fail compilation or produce unusable binaries. |
| `-I...` | Add include search paths. | Runtime headers, tensor ABI headers, kernel library headers, generated headers. | Missing or wrong include order can silently pick incompatible headers. |
| `-x musa` | Force the input file language to MUSA. | Build systems or files whose suffix does not imply MUSA. | Use only when the source should be compiled by the MUSA frontend. |
| `-DNDEBUG` | Define `NDEBUG` and disable C/C++ `assert`. | Release and performance builds. | Removes assertion checks; keep separate debug paths for correctness work. |
| `-D...` | Define a preprocessor macro. | Feature guards, generated-version headers, template dispatch, debug/release switches. | Keep macro values consistent across translation units. |
| `-Wno-switch-bool` | Suppress bool-switch warnings. | Generated or third-party code that intentionally switches on boolean-like values. | Warning-only; does not change codegen intent. |
| `-v` | Verbose compiler output. | Compiler-command debugging and build-system inspection. | Adds log noise; not a performance option. |
| `-fno-strict-aliasing` | Disable strict-aliasing assumptions. | Kernels or headers that reinterpret buffers, use packed vector types, cast raw pointers, or share storage between logical views. | Can reduce some optimization opportunities, but avoids aliasing-related miscompiles. |
| `-fno-signed-zeros` | Allow the compiler to treat `+0.0` and `-0.0` as equivalent. | Numeric kernels where exact signed-zero behavior is irrelevant: tiled arithmetic, row-wise reductions, activation fusion, and most inference kernels. | Changes strict IEEE behavior for signed zero. |
| `-fmusa-flush-denormals-to-zero` | Flush subnormal floating-point values to zero. | Throughput kernels where tiny denormals are noise: reductions, scans, score transforms, fused elementwise math, and streaming kernels. | Changes numerical behavior for very small values. |
| `-mllvm` | Pass the next option through to the LLVM/MUSA backend. | Required prefix for the backend options below. | Keep it paired with the following backend option. |

## MTGPU Backend Options

These options are passed through with `-mllvm`, for example:

```text
-mllvm -mtgpu-if-convert=1
```

| Backend option | Function | Kernel scenario | Risk or note |
|---|---|---|---|
| `-mtgpu-load-cluster-mutation=1` | Enable load clustering or load instruction mutation. | Global-load-heavy kernels with contiguous or semi-contiguous reads: tiled matrix loads, vectorized copy/reorder, streaming row reads. | May be compiler-version-sensitive; verify assembly and latency. |
| `--num-dwords-of-load-in-mutation=64` | Control the load-mutation data granularity. | Large streaming loads or tiled loads where the compiler can group enough adjacent dwords. | Tune only with assembly/benchmark evidence. |
| `-misched=mtgpu-max-ilp` | Select an instruction scheduler biased toward higher ILP. | Kernels with independent math and memory instructions inside the same loop: scans, recurrences, indexed loops, small reductions. | Can increase register pressure or hurt latency on some shapes. |
| `-mtgpu-if-convert=1` | Enable if-conversion, converting branches into predicated/select-like code where profitable. | Masked kernels with cheap branch bodies: tail predicates, validity checks, boundary checks, and short mode branches. | Can execute more instructions on paths that would otherwise branch away. |
| `-mtgpu-tiny-offset-hint=1` | Hint that small-offset addressing patterns should be optimized. | Kernels indexing fixed small vectors, short tiles, compact metadata, or narrow reductions. | Backend-specific; measure before keeping. |
| `-misched-recompute-slotindex=1` | Recompute scheduler slot indexes during machine scheduling. | Schedule-sensitive loops where many instructions compete for issue slots: reductions, recurrent scans, indexed loops, state-update loops. | Internal scheduler knob; can change across MCC versions. |
| `-mtgpu-combine-fop-instr=1` | Combine floating-point operation instructions. | Math-heavy kernels with repeated FMA/add/mul sequences: row transforms, state updates, accumulation loops, fused activation. | Can alter instruction selection; inspect numerical and performance impact. |
| `-mtgpu-enable-postra-sched=0` | Disable post-register-allocation scheduling. | Use when post-RA scheduling causes spills, bad issue order, or regressions in register-pressure-heavy kernels such as long state-update loops. | Disabling can help or hurt depending on register allocation and shape. |
| `-mtgpu-opt-level=1` | Enable MTGPU backend optimization level 1. | Conservative first step for generated kernels, especially streaming, elementwise, or reduction kernels. | Low-risk starting point, but still compiler-stack dependent. |
| `-mtgpu-load-store-opt=1` | Enable load/store optimization. | Memory-bandwidth-bound kernels when assembly shows scalarized or poorly scheduled LSU: streaming transforms, elementwise, copy, quant/dequant, cache transforms. | Must verify emitted LSU width and latency. |
| `-mtgpu-fold-global-ldst=1` | Fold or combine global load/store patterns. | Kernels with repeated adjacent global loads/stores from simple address expressions. | Can be shape-dependent. |
| `-mtgpu-store-cluster-mutation=1` | Enable store clustering or store instruction mutation. | Store-heavy kernels: row-wise output, elementwise output, quantization output, cache writeback, scatter-free contiguous stores. | Check store width and ordering assumptions. |
| `-mtgpu-memory-sched-mutation=1` | Enable memory scheduling mutation. | Kernels bottlenecked by memory instruction order rather than math: streaming reads/writes, row-wise reductions, copy, layout transform, type conversion. | Can trade memory-level parallelism against register pressure. |

## Kernel-Pattern Guidance

Choose flags by matching the kernel's bottleneck and control-flow shape, not by
copying every flag from another operator.

| Kernel pattern | Symptoms | Flags to try | Why these match |
|---|---|---|---|
| Tiled matrix multiply or tiled accumulation | Hot loops load two or more tiles, many FMAs, performance depends on load grouping and instruction scheduling. | `-Od3`, `-fno-strict-aliasing`, `-fno-signed-zeros`, `-mllvm -mtgpu-load-cluster-mutation=1`, `-mllvm --num-dwords-of-load-in-mutation=64` | Load mutation can improve tile-read instruction shape while relaxed floating-point assumptions fit throughput kernels. |
| Tiled score-and-accumulate kernel | Tiled vector/matrix loads, row-wise score transform, reduction, masks, and value accumulation. | Common handwritten performance set; additionally test `-mtgpu-if-convert=1` only if mask branches are cheap and frequent. | Load clustering helps tile reads; signed-zero and alias flags fit numeric throughput kernels; if-conversion may reduce branch divergence for masks. |
| Small-batch indexed kernel | Low CTA count, short loops, metadata indexing, many small offsets and predicates. | `-O3`, `-mllvm -misched=mtgpu-max-ilp`, `-mllvm -mtgpu-if-convert=1`, `-mllvm -mtgpu-tiny-offset-hint=1`, `-mllvm -misched-recompute-slotindex=1` | These kernels are often scheduling and control-flow sensitive, with compact address arithmetic and limited parallelism. |
| Row-wise streaming reduction | Streaming input/output, simple per-row reductions, memory bandwidth or LSU width is limiting. | Start with `-fmusa-flush-denormals-to-zero`, `-fno-signed-zeros`, `-mllvm -mtgpu-opt-level=1`; try `load-store-opt`, `fold-global-ldst`, load/store cluster, and memory sched only after assembly shows memory issues. | These kernels need clean load/store code and can often ignore denormals/signed zero. |
| Elementwise or activation-style fusion | Simple contiguous reads/writes, scalar math, possible store bottleneck. | `-O3`, `-fno-signed-zeros`, `-fmusa-flush-denormals-to-zero`; for store-heavy cases test `-mtgpu-store-cluster-mutation=1` and `-mtgpu-memory-sched-mutation=1`. | Aggressive scalar optimization and relaxed floating semantics help math; store mutation helps when output traffic dominates. |
| Reduction or scan | Repeated add/max/state updates, lane/warp reductions, sensitive scheduling. | `-O3` or `-Od3`, `-mllvm -misched=mtgpu-max-ilp`, `-mllvm -misched-recompute-slotindex=1`, `-mllvm -mtgpu-combine-fop-instr=1` | ILP and FOP combination can improve hot recurrence/reduction loops, but watch registers. |
| Indexed or masked kernels | Index arrays, validity checks, irregular but bounded work, compact metadata. | `-fmusa-flush-denormals-to-zero`, `-fno-strict-aliasing`, `-fno-signed-zeros`, `-mllvm -mtgpu-tiny-offset-hint=1`, optionally `-mllvm -mtgpu-if-convert=1`. | Tiny-offset and if-conversion target metadata and predicate-heavy code. |
| Layout transform, copy, quant/dequant | Mostly global memory traffic, contiguous or vectorizable loads/stores. | `-mllvm -mtgpu-load-store-opt=1`, `-mllvm -mtgpu-fold-global-ldst=1`, `-mllvm -mtgpu-load-cluster-mutation=1`, `-mllvm -mtgpu-store-cluster-mutation=1` | These target LSU quality directly; validate vector width in assembly. |
| Register-pressure-heavy fused kernel | Occupancy falls, spills appear, or post-RA scheduling changes worsen code. | Try removing aggressive flags first; if post-RA scheduling is implicated, test `-mllvm -mtgpu-enable-postra-sched=0`. | More flags can increase pressure; disabling a late scheduler is a targeted workaround, not a default. |

## Generic Flag Sets By Kernel Shape

Use these as starting points. They are examples of coherent flag sets for a
kernel shape, not mandatory recipes.

### Normal JIT `.mu` Baseline

These are the common baseline flags for a normal JIT `.mu` build:

```text
-fPIC
-std=c++17
-O2
--offload-arch=mp_${ARCH}
-I...
```

### Handwritten Performance Kernel

Useful for manually written kernels with tiled loads, a stable hot loop, and
throughput-oriented floating-point math:

```text
-Od3
-O2
-DNDEBUG
-fno-strict-aliasing
-fno-signed-zeros
-mllvm -mtgpu-load-cluster-mutation=1
-mllvm --num-dwords-of-load-in-mutation=64
```

Start from the common JIT flags, then add this set and benchmark.

### Generated Or Third-Party Warning Suppression

When generated or third-party headers produce bool-switch warnings:

```text
-Wno-switch-bool
```

### Generated Tiled Kernel

```text
-Od3
-DNDEBUG
-fno-strict-aliasing
-fno-signed-zeros
-mllvm -mtgpu-load-cluster-mutation=1
-mllvm --num-dwords-of-load-in-mutation=64
```

Use when the baseline already supplies `-O2`, and the generated device source
has a stable tiled hot loop.

### Template-Heavy Generated Kernel

```text
-Od3
-O2
-DNDEBUG
-fno-strict-aliasing
-mllvm -mtgpu-load-cluster-mutation=1
-mllvm --num-dwords-of-load-in-mutation=64
-std=c++17
```

Use for generated kernels that rely heavily on C++17 templates and tiled
arithmetic helpers.

### Long Masked Loop

```text
-fmusa-flush-denormals-to-zero
-fno-signed-zeros
-fno-strict-aliasing
-mllvm -misched=mtgpu-max-ilp
-mllvm -mtgpu-if-convert=1
-mllvm -mtgpu-tiny-offset-hint=1
-mllvm -misched-recompute-slotindex=1
-mllvm -mtgpu-combine-fop-instr=1
```

Use for long hot loops with masks, validity checks, metadata loads, and
memory/math interleaving.

### Short Indexed Loop

```text
-fmusa-flush-denormals-to-zero
-fno-signed-zeros
-fno-strict-aliasing
-mllvm -misched=mtgpu-max-ilp
-mllvm -mtgpu-tiny-offset-hint=1
-mllvm -misched-recompute-slotindex=1
-mllvm -mtgpu-combine-fop-instr=1
```

Use for short indexed loops where scheduler and tiny-offset hints help, but
if-conversion is not clearly beneficial.

### Streaming Row-Reduction Profiles

`opt1`:

```text
-fmusa-flush-denormals-to-zero
-fno-signed-zeros
-mllvm -mtgpu-opt-level=1
```

`ls`:

```text
-fmusa-flush-denormals-to-zero
-fno-signed-zeros
-mllvm -mtgpu-opt-level=1
-mllvm -mtgpu-load-store-opt=1
-mllvm -mtgpu-fold-global-ldst=1
-mllvm -mtgpu-load-cluster-mutation=1
-mllvm -mtgpu-store-cluster-mutation=1
-mllvm -mtgpu-memory-sched-mutation=1
```

Use `opt1` as a low-risk measured profile. Try `ls` when assembly shows poor
load/store quality or memory scheduling is the bottleneck.

### Scan Or State-Update Kernels

Generic recurrent or scan-like update:

```text
-Od3
-fno-signed-zeros
-mllvm -mtgpu-if-convert=1
-mllvm -misched=mtgpu-max-ilp
-mllvm -mtgpu-tiny-offset-hint=1
-mllvm -misched-recompute-slotindex=1
-mllvm -mtgpu-combine-fop-instr=1
```

For longer state-update loops, add denormal flushing and test disabling post-RA
scheduling:

```text
-Od3
-fno-signed-zeros
-fmusa-flush-denormals-to-zero
-mllvm -mtgpu-if-convert=1
-mllvm -misched=mtgpu-max-ilp
-mllvm -mtgpu-tiny-offset-hint=1
-mllvm -mtgpu-enable-postra-sched=0
-mllvm -misched-recompute-slotindex=1
-mllvm -mtgpu-combine-fop-instr=1
```

For simple row-wise reduction, `-O3` can be a good candidate:

```text
-O3
-fno-signed-zeros
-mllvm -mtgpu-if-convert=1
-mllvm -misched=mtgpu-max-ilp
-mllvm -mtgpu-tiny-offset-hint=1
-mllvm -misched-recompute-slotindex=1
-mllvm -mtgpu-combine-fop-instr=1
```

For short indexed update loops:

```text
-O3
-mllvm -misched=mtgpu-max-ilp
-mllvm -mtgpu-if-convert=1
-mllvm -mtgpu-tiny-offset-hint=1
-mllvm -misched-recompute-slotindex=1
```

### CMake Or Template Library Builds

```text
-std=c++17
-fno-strict-aliasing
-Od3
-DMUTLASS_VERSIONS_GENERATED
--offload-arch=mp_${ARCH}
-x musa
-fPIC
-v
```

`-v` is only for verbose compiler output.

## Environment Controls

| Control | Function | Scenario |
|---|---|---|
| `mcc` path override | Select the compiler executable used by the build system. | Compare compilers or use a non-default MUSA toolkit. |
| target arch list | Control the `--offload-arch=mp_${ARCH}` values. | Build without a visible device or force a specific target such as `3.1`. |
| extra device flags | Append MUSA compiler flags to JIT `.mu` builds. | Short experiments without editing source files. |
| extra host C++ flags | Append host compiler flags. | Host wrapper or extension compilation experiments. |
| extra linker flags | Append linker flags. | Library path or extra dependency experiments. |
| build parallelism | Control concurrent compiler jobs. | Avoid overloading shared build machines or reduce memory pressure. |

## Practical Selection Rules

1. For a normal MUSA extension kernel, start with:

   ```text
   -fPIC -std=c++17 -O2 --offload-arch=mp_${ARCH}
   ```

2. For common handwritten performance kernels, add:

   ```text
   -Od3 -DNDEBUG -fno-strict-aliasing -fno-signed-zeros
   -mllvm -mtgpu-load-cluster-mutation=1
   -mllvm --num-dwords-of-load-in-mutation=64
   ```

3. For generated memory/math kernels, try only one profile at a time. Choose by
   matching the kernel pattern, then benchmark and inspect assembly.

4. Do not cargo-cult all backend flags together. Many flags target specific
   scheduler, load/store, branch, or denormal behavior and may regress other
   kernels.

5. Whenever a performance claim depends on a backend flag, record the MCC/MUSA
   version, kernel shape, correctness result, latency, and a short assembly or
   generated-code observation.

# Assembly And Generated-Code Audit

Inspect generated MUSA source, MusaAsm, or compiler assembly when performance
depends on vectorization, cache hints, barriers, warp intrinsics, SQMMA/TME, or
address math.

## Audit Checklist

| Item | Good sign | Warning sign |
|---|---|---|
| LSU width | Expected `B64` or `B128` on vector paths | Scalar `B16`/`B32` in hot loop |
| Store/load hints | Expected cache/SLC fields on targeted instructions | Hint missing or applied to wrong instruction |
| Address math | Constants, shifts, simple adds in inner loop | Repeated divisions, modulo, 64-bit multiply-add |
| Register pressure | No local-memory spills | Large local arrays or spill loads/stores |
| Barriers | Minimal required syncs | Sync inside hot loop without data dependency |
| Branches | Fast path has few runtime conditions | Rare feature branches remain in common hot path |
| Expensive math | Only appears for true activation/softmax/sigmoid work | Accidental transcendental ops in bandwidth path |

## Inline Assembly Policy

| Rule | Reason |
|---|---|
| Guard architecture-specific assembly | MUSA assembly syntax and cache fields are not guaranteed portable |
| Provide a normal C++ fallback | Keeps other architectures and compile modes correct |
| Add inline assembly only after measurement | Avoids unproven complexity |
| Re-run correctness and benchmark after assembly changes | Cache hints and vector stores can expose alignment or ordering issues |
| Keep assembly scoped to a helper | Makes future replacement or arch gating easier |

## What To Look For By Bottleneck

| Bottleneck | Inspect |
|---|---|
| Memory bandwidth | LSU width, coalescing, cache hints, extra reloads |
| Launch/latency | Number of kernels, CTA count, small-shape overhead |
| Sync-bound | Barrier count and placement |
| Instruction-bound | Transcendentals, conversion sequences, integer address math |
| Occupancy-bound | registers, shared memory, block size, active CTAs |

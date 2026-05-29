# Optimization Levers

Use one lever at a time and keep only measured wins on the target workload.

| Lever | When to try | Risk |
|---|---|---|
| Launch shape | Independent axes are serialized or CTA count is too low | Too many CTAs can increase overhead |
| Thread count | Reductions or memory streaming underperform | Higher occupancy can increase sync/register pressure |
| Vectorized load/store | Data is contiguous and memory-bound | Assembly may still scalarize; alignment bugs |
| Subwarp reductions | Small independent groups | Lane masks and group ownership must be exact |
| CTA reductions | Larger groups need cross-warp reduction | Shared memory and barriers can dominate |
| Shared-memory cache | Data is reread within a CTA | Can reduce occupancy or add bank conflicts |
| Register cache | Small tiles or decode-like shapes | Spills can erase gains |
| Tiling multiple rows/groups | Small hidden/group count underutilizes CTAs | More complicated tail handling |
| Branch split | Rare options slow the common path | More dispatch branches to maintain |
| Fusion | Removes global memory traffic | Adds registers, math, and validation complexity |
| Compile-time constants | Common dimensions or flags dominate | Code size and specialization bloat |
| Cache hints | Streaming copy/store or cache pollution suspected | Shape-dependent; must measure |
| Inline assembly | Compiler cannot express needed memory/cache behavior | Arch-specific and harder to maintain |

## Change-One-Thing Loop

1. Measure baseline.
2. Form one hypothesis.
3. Change one parameter or one structural feature.
4. Re-run correctness.
5. Re-measure target shapes.
6. Keep, revert, or record as rejected.

## Bottleneck Hints

| Observation | Likely bottleneck |
|---|---|
| Good large-shape bandwidth, bad small `m` | launch latency or CTA setup |
| Low LSU width in assembly | vectorization did not happen |
| Many syncs in hot loop | synchronization-bound |
| `exp`/`tanh` dominates | instruction-bound activation or softmax |
| Extra global reloads after reduction | missing cache/register reuse |
| Occupancy low with high registers | register pressure |
| Occupancy low with high shared memory | shared-memory footprint |

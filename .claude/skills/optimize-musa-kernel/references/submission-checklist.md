# Submission Checklist

Before submitting a MUSA kernel change:

| Item | Required |
|---|---|
| Operator contract documented | yes |
| Correctness against intended reference | yes |
| Tiny and target shape tests | yes |
| Dtype/layout variants covered by dispatch | yes |
| Benchmark table with comparator | yes |
| Bandwidth/TFLOPS formula included | yes |
| Cold-cache measurement for bandwidth claims | yes |
| Generated-code or assembly audit for low-level claims | yes |
| Fast-path guards explicit | yes |
| Fallback path correct | yes |
| No graph-unsafe host sync or torch fallback | yes |
| Rejected non-obvious attempts recorded | yes |
| Unrelated files left untouched | yes |

## Final Report Shape

Lead with the target-shape result:

```text
Kernel:
Comparator:
Target shape:
Correctness:
Best latency/BW:
Main accepted change:
Rejected attempts:
Remaining bottleneck:
Files changed:
Validation commands:
```

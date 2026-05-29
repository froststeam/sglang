# Debugging And Correctness

## Failure Classifier

| Category | Symptoms | First move |
|---|---|---|
| Build/codegen | Python traceback, TVM/TIR error, MUSA compile error | Reduce to minimal compile case and inspect generated source |
| Runtime crash | Launch failure, illegal access, timeout | Run tiny shape, check masks and global memory accesses |
| Hang/deadlock | Larger shapes never return | Audit barriers, waits, parity and producer/consumer ownership |
| Wrong result | Zeros, garbage, NaN/inf, large mismatch | Compare tiny-shape intermediates against reference |
| Numerical drift | Small tolerance failure | Check accumulation dtype, operation order, rounding and reference conventions |

## Correctness Rules

- Validate the reference independently.
- Start with deterministic tiny shapes.
- Test aligned and tail shapes.
- Test all dispatch dtype/layout variants that the new path claims to support.
- Check optional arguments, masks, invalid rows and empty inputs explicitly.
- Validate auxiliary outputs separately from main outputs.
- For fwd/bwd kernels, test each gradient tensor independently.
- Avoid graph-unsafe host synchronization in runtime code.

## Common Bugs

| Bug | Symptom |
|---|---|
| Missing tail guard | Illegal access or wrong last block |
| Wrong stride/layout mapping | Correct contiguous case but wrong strided case |
| Incorrect mutation semantics | Reference output matches, but in-place tensor is wrong |
| Bad reduction identity | NaN/incorrect result on all-invalid or empty rows |
| Uninitialized accumulator | Random mismatches across runs |
| Wrong dtype conversion | Large drift in fp16/bf16/fp8 paths |
| Barrier mismatch | Kernel hang |
| Host sync in runtime path | MUSA graph capture failure or severe latency regression |

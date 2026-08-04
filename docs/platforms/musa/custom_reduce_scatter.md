# MUSA Custom Reduce-Scatter d3

SGLang can use the node-local MUSA direct reduce-scatter d3 kernel for the
`attention_tp` group. The d3 implementation is the chunk-aware rank ordering
from commit `e53bcb9db3c6bc78877da583efa5f6c1aa4de4c9`.

## Gating

The path has one dedicated enable gate, disabled by default:

```bash
export SGLANG_MUSA_USE_JIT_REDUCE_SCATTER=1
```

It is independent of custom all-gather:

```bash
export SGLANG_MUSA_USE_JIT_ALL_GATHER=0
export SGLANG_MUSA_USE_JIT_REDUCE_SCATTER=1
```

No mode selector or custom-all-reduce compatibility gate is used. In
particular, this path does not read `SGLANG_MUSA_JIT_REDUCE_SCATTER_MODE`,
`SGLANG_MUSA_CUSTOM_RS_MCCL_COMPAT`, or
`SGLANG_MUSA_USE_CUSTOM_ALLREDUCE_RS`.

## Runtime Scope and Fallback

The d3 path is selected only when all of these conditions hold:

- the process runs on MUSA;
- the process group is node-local and named `attention_tp`;
- the group world size is 2, 4, 6, or 8;
- input and output are contiguous fp16, bf16, or fp32 tensors on the same device;
- the input shape is the output shape multiplied by the world size on dimension 0;
- input and output pointers are 16-byte aligned and the input fits the IPC
  staging buffer.

Unsupported calls fall back to the existing PyNccl or torch.distributed path.
Eager execution and the first graph-capture attempt use the fixed IPC staging
buffer. During graph capture, SGLang registers the stable input addresses and
recaptures the graph; replay then uses the registered d3 kernel to read peer
inputs directly and avoids the full-input staging copy. If graph input
registration is disabled, the staging path remains the graph-safe fallback.

## Tuning

The following variables tune allocation or compilation but do not enable the
path:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `SGLANG_CUSTOM_RS_MAX_SIZE_BYTES` | `536870912` | Maximum input size and per-rank IPC staging-buffer size. |
| `SGLANG_CUSTOM_RS_THREADS` | `512` | Kernel threads per block. |
| `SGLANG_CUSTOM_RS_BLOCKS` | `80` for world size 4/8, otherwise `56` | Default kernel block limit. |
| `SGLANG_CUSTOM_RS_MAX_BLOCKS` | at least `120` | Signal metadata and launch upper bound. |
| `SGLANG_CUSTOM_RS_DYNAMIC_BLOCKS` | `1` | Enable the size-dependent block cap. |
| `SGLANG_MUSA_CUSTOM_RS_GRAPH_REGISTERED_INPUT` | `1` | Use registered peer input addresses for graph-captured d3 reduce-scatter. Set to `0` to retain the staging path. |

For compatibility with the implementation where reduce-scatter lived inside
the custom all-reduce communicator, `SGLANG_MUSA_CUSTOM_AR_GRAPH_REGISTERED_INPUT`
is also honored when the RS-specific variable is unset.

## Test

On a MUSA host with eight devices (the 8-rank case exercises the d3 chunked
schedule directly):

```bash
pytest -q test/srt/musa/test_musa_custom_reduce_scatter.py
```

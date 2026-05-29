# CUDA/Triton To MUSA Porting Notes

## Porting Checklist

| Topic | CUDA/Triton habit | MUSA habit |
|---|---|---|
| Runtime | CUDA extension or Triton launch | MUSA runtime, TVM FFI loader, or TileLang `target="musa"` |
| Device | CUDA tensor/device checks | MUSA/Torch-MUSA device checks |
| Stream | CUDA stream | `musaStream_t` or runtime-provided stream |
| FP16/BF16 | CUDA half/bfloat16 types | MUSA half and MUSA bf16 types |
| FP8 | CUDA/Triton fp8 casts | MUSA fp8 types and conversion intrinsics |
| Cache hints | PTX/Triton modifiers | MUSA intrinsics or arch-guarded assembly |
| Assembly audit | PTX/SASS | generated MUSA code, MusaAsm, or compiler assembly |
| Graph safety | No host sync in CUDA graph | Same rule for MUSA graph capture |

## Rules

- Do not assume CUDA type names, intrinsics, or cache modifiers compile on MUSA.
- Keep layout and mutation semantics identical to the existing SGLang API.
- Replace CUDA-only helpers with local MUSA abstractions when available.
- Gate architecture-specific behavior.
- Validate numerical conventions after changing dtype conversion or packing.
- Inspect generated code when porting vectorized memory or warp primitive logic.

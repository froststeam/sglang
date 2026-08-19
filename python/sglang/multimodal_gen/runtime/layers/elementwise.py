import torch

from sglang.jit_kernel.diffusion.triton.scale_shift import fuse_scale_shift_kernel
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.platforms import current_platform


_is_musa = current_platform.is_musa()

if _is_musa:
    from sglang.srt.hardware_backend.musa.jit_kernel.csrc.diffusion import (
        can_use_musa_mul_add,
        musa_mul_add,
    )


class MulAdd(CustomOp):
    """
    Fuse elementwise mul and add
    Input: a, b, c, OptionalInt[k]
    Output: a * (k + b) + c
    """

    def __init__(self, prefix: str = ""):
        super().__init__()

    def forward_native(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0
    ) -> torch.Tensor:
        # a.shape: [batch_size, seq_len, inner_dim]
        if b.dim() == 4:
            # b.shape: [batch_size, num_frames, 1, inner_dim]
            num_frames = b.shape[1]
            frame_seqlen = a.shape[1] // num_frames
            return c + (
                a.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (k + b)
            ).flatten(1, 2)
        else:
            # b.shape: [batch_size, 1, inner_dim]
            return c + a * (k + b)

    def forward_cuda(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0
    ):
        return fuse_scale_shift_kernel(a, b, c, scale_constant=k)

    def forward_musa(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0
    ):
        if (
            k == 0
            and a.ndim == b.ndim == c.ndim == 3
            and a.shape == c.shape
            and b.shape == (a.shape[0], 1, a.shape[-1])
            and a.is_contiguous()
            and c.is_contiguous()
            and a.dtype == b.dtype == c.dtype
            and can_use_musa_mul_add(a.shape[-1], a.dtype)
        ):
            return musa_mul_add(a, b, c)
        return self.forward_native(a, b, c, k=k)

    def forward_xpu(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0
    ):
        return self.forward_native(a, b, c, k=k)

    def forward_npu(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0
    ):
        from sgl_kernel_npu.norm.scale_shift import fused_scale_shift

        return fused_scale_shift(a, b, c, scale_constant=k)

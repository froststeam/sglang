import torch

from ..kernels.hisparse_kernels import _tilelang_hisparse_offload_to_host_kernel


def hisparse_offload_to_host_musa(
    gpu_ptrs: torch.Tensor,
    cpu_ptrs: torch.Tensor,
    gpu_indices: torch.Tensor,
    cpu_indices: torch.Tensor,
) -> None:
    if gpu_ptrs.device.type != "musa" or cpu_ptrs.device.type != "musa":
        raise NotImplementedError("DeepSeekV4 MUSA hisparse_offload_to_host expects MUSA pointer tables")
    if gpu_indices.device != gpu_ptrs.device or cpu_indices.device != gpu_ptrs.device:
        raise NotImplementedError("DeepSeekV4 MUSA hisparse_offload_to_host expects indices on the pointer-table device")
    if gpu_ptrs.dtype != torch.uint64 or cpu_ptrs.dtype != torch.uint64:
        raise NotImplementedError("DeepSeekV4 MUSA hisparse_offload_to_host expects uint64 pointer tables")
    if gpu_indices.dtype != torch.int64 or cpu_indices.dtype != torch.int64:
        raise NotImplementedError("DeepSeekV4 MUSA hisparse_offload_to_host expects int64 indices")
    if gpu_ptrs.dim() != 1 or cpu_ptrs.dim() != 1 or gpu_ptrs.shape != cpu_ptrs.shape:
        raise NotImplementedError("DeepSeekV4 MUSA hisparse_offload_to_host expects matching 1D pointer tables")
    if gpu_indices.dim() != 1 or cpu_indices.dim() != 1 or gpu_indices.shape != cpu_indices.shape:
        raise NotImplementedError("DeepSeekV4 MUSA hisparse_offload_to_host expects matching 1D indices")
    if gpu_indices.numel() == 0 or gpu_ptrs.numel() == 0:
        return
    if not gpu_ptrs.is_contiguous() or not cpu_ptrs.is_contiguous():
        raise NotImplementedError("DeepSeekV4 MUSA hisparse_offload_to_host expects contiguous pointer tables")
    if not gpu_indices.is_contiguous() or not cpu_indices.is_contiguous():
        raise NotImplementedError("DeepSeekV4 MUSA hisparse_offload_to_host expects contiguous indices")

    kernel = _tilelang_hisparse_offload_to_host_kernel()
    kernel(gpu_ptrs, cpu_ptrs, gpu_indices, cpu_indices)


__all__ = [
    "hisparse_offload_to_host_musa",
]

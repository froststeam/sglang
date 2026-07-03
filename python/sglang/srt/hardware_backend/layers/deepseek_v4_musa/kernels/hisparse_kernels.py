from functools import lru_cache

from .kernel_common import _tilelang_jit, _tilelang_musa_aggressive_pass_configs


@lru_cache(maxsize=None)
def _tilelang_hisparse_offload_to_host_kernel(threads: int = 128):
    import tilelang
    import tilelang.language as T

    num_items = T.dynamic("num_items")
    num_layers = T.dynamic("num_layers")

    cpu_item_u64 = 73
    gpu_value_u64 = 72
    gpu_page_size = 64
    gpu_page_bits = 6
    gpu_page_u64 = 4680
    gpu_scale_offset_u64 = 4608
    copy_u64 = cpu_item_u64
    copy_blocks = (copy_u64 + threads - 1) // threads
    max_u64 = 1 << 60

    @_tilelang_jit(
        tilelang,
        f"dsv4_hisparse_offload_to_host_t{threads}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(tilelang, disable_index_promotion=False),
    )
    def hisparse_offload_to_host_kernel(
        gpu_ptrs: T.Tensor[(num_layers,), T.uint64],
        cpu_ptrs: T.Tensor[(num_layers,), T.uint64],
        gpu_indices: T.Tensor[(num_items,), T.int64],
        cpu_indices: T.Tensor[(num_items,), T.int64],
    ):
        with T.Kernel(num_items, num_layers, copy_blocks, threads=threads) as (item_id, layer_id, copy_block):
            gpu_base = gpu_ptrs[layer_id]
            cpu_base = cpu_ptrs[layer_id]
            gpu_cache = T.make_tensor_from_addr(gpu_base, (max_u64,), dtype=T.uint64)
            cpu_cache = T.make_tensor_from_addr(cpu_base, (max_u64,), dtype=T.uint64)
            gpu_index = T.cast(gpu_indices[item_id], T.int64)
            cpu_index = T.cast(cpu_indices[item_id], T.int64)
            gpu_page = gpu_index >> gpu_page_bits
            gpu_page_offset = gpu_index & (gpu_page_size - 1)
            gpu_page_base = gpu_page * gpu_page_u64
            cpu_item_base = cpu_index * cpu_item_u64

            for elem in T.Parallel(threads):
                copy_id = copy_block * threads + elem
                if copy_id < copy_u64:
                    src_offset = T.alloc_local((1,), dtype=T.int64)
                    if copy_id < gpu_value_u64:
                        src_offset[0] = gpu_page_base + gpu_page_offset * gpu_value_u64 + copy_id
                    else:
                        src_offset[0] = gpu_page_base + gpu_scale_offset_u64 + gpu_page_offset
                    cpu_cache[cpu_item_base + copy_id] = gpu_cache[src_offset[0]]

    return hisparse_offload_to_host_kernel


__all__ = [
    "_tilelang_hisparse_offload_to_host_kernel",
]

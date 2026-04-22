import time

import sgl_kernel
import torch

from sglang.srt.utils import get_device


def measure_time(fn, warmup=5, trials=20):
    """Measure average time (s) of fn() over several trials."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)


def torch_top_k_top_p_sampling_from_logits(logits, k, p):
    """Reference PyTorch implementation for comparison."""
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # top-p mask
    mask = cumulative_probs > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_probs[mask] = 0.0

    # top-k mask
    if k < sorted_probs.size(-1):
        sorted_probs[..., k:] = 0.0

    # renormalize
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(sorted_probs, 1).squeeze(-1)
    return sorted_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


def benchmark_sampling(
    batch_size=512, vocab_size=32000, k=100, p=0.9, dtype=torch.float16
):
    device = get_device(0)
    logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype) * 5

    # FlashInfer
    avg_flashinfer = measure_time(
        lambda: sgl_kernel.sampling.top_k_top_p_sampling_from_logits(
            logits, k, p, filter_apply_order="joint"
        )
    )

    # Torch baseline
    avg_torch = measure_time(
        lambda: torch_top_k_top_p_sampling_from_logits(logits, k, p)
    )

    print(
        f"batch={batch_size:4d} vocab={vocab_size:6d} "
        f"k={k:4d} p={p:4.2f} "
        f"flashinfer={avg_flashinfer*1000:.3f}ms "
        f"torch={avg_torch*1000:.3f}ms "
        f"speedup={avg_torch/avg_flashinfer:5.2f}x"
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    for batch in [1, 64, 512, 1024]:
        for vocab in [8192, 32000, 128256]:
            for k in [50, 100]:
                for p in [0.7, 0.9]:
                    benchmark_sampling(batch, vocab, k, p)

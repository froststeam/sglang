import pytest
import torch
from sgl_kernel import moe_fused_gate

from sglang.srt.layers.moe.topk import biased_grouped_topk
from sglang.srt.utils import get_device, is_musa

_is_musa = is_musa()
_is_mate = False

if _is_musa:
    try:
        from mate import moe_fused_gate

        _is_mate = True
    except ImportError:
        print(
            "mate is MUSA specific kernel library. Please make sure mate is installed on your MUSA device."
        )


def get_cuda_params():
    return [
        (128, 4, 2, 4),
        (160, 1, 1, 8),
        (256, 1, 1, 8),
        (256, 4, 2, 8),
    ]


def get_musa_params():
    return [
        (128, 4, 2, 4),
        (256, 8, 4, 8),
        (512, 16, 8, 16),
        (160, 1, 1, 8),
        (256, 1, 1, 8),
        (384, 1, 1, 8),
    ]


@pytest.mark.parametrize(
    "seq_length",
    list(range(1, 10))
    + [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
)
@pytest.mark.parametrize(
    "params", get_cuda_params() if not _is_musa else get_musa_params()
)
@pytest.mark.parametrize("num_fused_shared_experts", [0])
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
def test_moe_fused_gate_combined(
    seq_length, params, num_fused_shared_experts, apply_routed_scaling_factor_on_output
):
    num_experts, num_expert_group, topk_group, topk = params
    dtype = torch.float32

    torch.manual_seed(seq_length)
    tensor = torch.rand((seq_length, num_experts), dtype=dtype, device=get_device())
    scores = tensor.clone()
    bias = torch.rand(num_experts, dtype=dtype, device=get_device())
    topk = topk + num_fused_shared_experts

    kwargs = {}
    if _is_musa and _is_mate:
        kwargs["renormalize"] = True

    output, indices = moe_fused_gate(
        tensor,
        bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        **kwargs,
    )
    ref_output, ref_indices = biased_grouped_topk(
        scores,
        scores,
        bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    # When num_fused_shared_experts > 0, ignore the comparison of the last topk dimension
    if num_fused_shared_experts > 0:
        original_indices = indices.clone()
        original_ref_indices = ref_indices.clone()

        indices = indices[:, :-1]
        ref_indices = ref_indices[:, :-1]

        valid_min = num_experts
        valid_max = num_experts + num_fused_shared_experts
        shared_indices = original_indices[:, -1]
        shared_ref_indices = original_ref_indices[:, -1]
        if shared_indices is not None:
            assert torch.all(
                (shared_indices >= valid_min) & (shared_indices < valid_max)
            ), f"Shared expert indices out of range: found values outside [{valid_min}, {valid_max})"
        if shared_ref_indices is not None:
            assert torch.all(
                (shared_ref_indices >= valid_min) & (shared_ref_indices < valid_max)
            ), f"Shared expert reference indices out of range: found values outside [{valid_min}, {valid_max})"

    idx_check = torch.allclose(
        ref_indices.sort()[0].to(torch.int32),
        indices.sort()[0].to(torch.int32),
        rtol=1e-04,
        atol=1e-05,
    )
    output_check = torch.allclose(
        ref_output.sort()[0].to(torch.float32),
        output.sort()[0].to(torch.float32),
        rtol=1e-02,
        atol=1e-03,
    )

    assert idx_check, (
        f"Indices mismatch at seq_length {seq_length}, dtype {dtype}, "
        f"params {params}, num_fused_shared_experts {num_fused_shared_experts}"
    )
    assert output_check, (
        f"Output mismatch at seq_length {seq_length}, dtype {dtype}, "
        f"params {params}, num_fused_shared_experts {num_fused_shared_experts}"
    )


if __name__ == "__main__":
    pytest.main([__file__])

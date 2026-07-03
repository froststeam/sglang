import unittest

import torch

from sglang.srt.model_executor.cuda_graph_runner import DecodeInputBuffers


def _create_buffers(**kwargs):
    params = dict(
        device=torch.device("cpu"),
        max_bs=4,
        max_num_token=4,
        hidden_size=16,
        pp_proxy_hidden_size=None,
        pp_proxy_tensor_names=None,
        vocab_size=128,
        dtype=torch.float32,
        dp_size=1,
        pp_size=4,
        is_encoder_decoder=False,
        require_mlp_tp_gather=False,
        seq_len_fill_value=1,
        encoder_len_fill_value=0,
        num_tokens_per_bs=1,
        cache_loc_dtype=torch.int64,
        enable_mamba_track=False,
    )
    params.update(kwargs)
    return DecodeInputBuffers.create(**params)


class TestCudaGraphPPProxyTensors(unittest.TestCase):
    def test_default_pp_proxy_tensors_match_existing_layout(self):
        buffers = _create_buffers()

        self.assertEqual(set(buffers.pp_proxy_tensors), {"hidden_states", "residual"})
        self.assertEqual(buffers.pp_proxy_tensors["hidden_states"].shape, (4, 16))
        self.assertEqual(buffers.pp_proxy_tensors["residual"].shape, (4, 16))

    def test_model_declared_pp_proxy_shape_and_keys(self):
        buffers = _create_buffers(
            pp_proxy_hidden_size=64,
            pp_proxy_tensor_names=("hidden_states",),
        )

        self.assertEqual(set(buffers.pp_proxy_tensors), {"hidden_states"})
        self.assertEqual(buffers.pp_proxy_tensors["hidden_states"].shape, (4, 64))
        self.assertNotIn("residual", buffers.pp_proxy_tensors)

    def test_non_pp_keeps_proxy_tensors_disabled(self):
        buffers = _create_buffers(
            pp_size=1,
            pp_proxy_hidden_size=64,
            pp_proxy_tensor_names=("hidden_states",),
        )

        self.assertIsNone(buffers.pp_proxy_tensors)


if __name__ == "__main__":
    unittest.main()

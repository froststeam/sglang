import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import dp_attention


class TestZeroDpPaddingTokensAfterAttnTpGather(unittest.TestCase):
    def _forward_batch(self, original_global_num_tokens, global_num_tokens):
        return SimpleNamespace(
            original_global_num_tokens_cpu=original_global_num_tokens,
            global_num_tokens_cpu=global_num_tokens,
        )

    def test_zeros_padding_for_global_dp_counts_on_rank_zero(self):
        hidden_states = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        forward_batch = self._forward_batch([2, 4], [4, 4])

        with patch.object(dp_attention, "get_attention_dp_rank", return_value=0):
            dp_attention.zero_dp_padding_tokens_after_attn_tp_gather(
                hidden_states, forward_batch
            )

        torch.testing.assert_close(
            hidden_states,
            torch.tensor([[0.0, 1.0], [2.0, 3.0], [0.0, 0.0], [0.0, 0.0]]),
        )

    def test_zeros_padding_for_global_dp_counts(self):
        hidden_states = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        forward_batch = self._forward_batch([2, 3], [2, 4])

        with patch.object(dp_attention, "get_attention_dp_rank", return_value=1):
            dp_attention.zero_dp_padding_tokens_after_attn_tp_gather(
                hidden_states, forward_batch
            )

        torch.testing.assert_close(
            hidden_states,
            torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [0.0, 0.0]]),
        )

    def test_zeros_padding_for_local_dp_count_on_nonzero_rank(self):
        hidden_states = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        forward_batch = self._forward_batch([3], [4])

        with patch.object(dp_attention, "get_attention_dp_rank", return_value=1):
            dp_attention.zero_dp_padding_tokens_after_attn_tp_gather(
                hidden_states, forward_batch
            )

        torch.testing.assert_close(
            hidden_states,
            torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [0.0, 0.0]]),
        )

    def test_preserves_hidden_states_without_padding(self):
        hidden_states = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        forward_batch = self._forward_batch([4], [4])

        with patch.object(dp_attention, "get_attention_dp_rank", return_value=1):
            dp_attention.zero_dp_padding_tokens_after_attn_tp_gather(
                hidden_states, forward_batch
            )

        torch.testing.assert_close(
            hidden_states, torch.arange(8, dtype=torch.float32).reshape(4, 2)
        )


if __name__ == "__main__":
    unittest.main()

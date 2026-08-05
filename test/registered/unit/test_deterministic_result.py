import unittest

from sglang.test.test_deterministic import (
    compare_response_sequence,
    compare_responses,
    deterministic_test_exit_code,
    deterministic_test_passed,
    sample_prefix_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def make_response(text="answer", output_ids=None, logprobs=None):
    output_ids = [10, 11] if output_ids is None else output_ids
    logprobs = (
        [[-0.1, output_ids[0], "a"], [-0.2, output_ids[1], "b"]]
        if logprobs is None
        else logprobs
    )
    return {
        "text": text,
        "output_ids": output_ids,
        "meta_info": {"output_token_logprobs": logprobs},
    }


class TestDeterministicResult(unittest.TestCase):
    def test_prefix_sampling_covers_every_prompt_class(self):
        for batch_size in range(4, 9):
            self.assertEqual(
                set(sample_prefix_indices(batch_size, num_prompts=4)),
                {0, 1, 2, 3},
            )

    def test_matching_response_passes(self):
        match, mismatches = compare_responses(
            make_response(), make_response(), compare_exact_logprobs=True
        )
        self.assertTrue(match)
        self.assertEqual(mismatches, [])

    def test_text_mismatch_fails(self):
        match, mismatches = compare_responses(
            make_response(), make_response(text="different")
        )
        self.assertFalse(match)
        self.assertTrue(any("Text mismatch" in item for item in mismatches))

    def test_token_mismatch_fails_even_when_text_matches(self):
        match, mismatches = compare_responses(
            make_response(), make_response(output_ids=[10, 12])
        )
        self.assertFalse(match)
        self.assertTrue(any("token IDs mismatch" in item for item in mismatches))

    def test_exact_logprob_mismatch_fails(self):
        match, mismatches = compare_responses(
            make_response(),
            make_response(logprobs=[[-0.1, 10, "a"], [-0.2000001, 11, "b"]]),
            compare_exact_logprobs=True,
        )
        self.assertFalse(match)
        self.assertTrue(any("Logprob mismatch" in item for item in mismatches))

    def test_logprob_token_id_mismatch_fails(self):
        match, mismatches = compare_responses(
            make_response(),
            make_response(logprobs=[[-0.1, 10, "a"], [-0.2, 12, "b"]]),
            compare_exact_logprobs=True,
        )
        self.assertFalse(match)
        self.assertTrue(any("Token ID mismatch" in item for item in mismatches))

    def test_result_summary_controls_exit_status(self):
        self.assertTrue(deterministic_test_passed([1, 1]))
        self.assertFalse(deterministic_test_passed([1, 0]))
        self.assertFalse(deterministic_test_passed([]))
        self.assertEqual(deterministic_test_exit_code([1, 1]), 0)
        self.assertEqual(deterministic_test_exit_code([1, 0]), 1)
        self.assertEqual(deterministic_test_exit_code([]), 1)

    def test_non_first_batch_response_mismatch_fails(self):
        responses = [
            make_response(),
            make_response(),
            make_response(output_ids=[10, 12]),
        ]

        match, mismatched_responses = compare_response_sequence(responses)

        self.assertFalse(match)
        self.assertEqual(mismatched_responses[0][0], 3)
        self.assertTrue(
            any(
                "token IDs mismatch" in item
                for item in mismatched_responses[0][1]
            )
        )

    def test_empty_response_sequence_fails(self):
        match, mismatched_responses = compare_response_sequence([])
        self.assertFalse(match)
        self.assertEqual(mismatched_responses, [(0, ["No responses returned"])])


if __name__ == "__main__":
    unittest.main()

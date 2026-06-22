import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def register_cpu_ci(*args, **kwargs):
    return None

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class DummyEval:
    def __init__(self):
        self.sampler = None

    def __call__(self, sampler):
        self.sampler = sampler
        return object()


class FakeCompletionSampler:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model")
        self._completion_tokens = []
        FakeCompletionSampler.instances.append(self)


class RunEvalOnceTest(unittest.TestCase):
    def setUp(self):
        FakeCompletionSampler.instances = []

    def _load_run_eval_module(self):
        run_eval_path = (
            Path(__file__).parents[3] / "python" / "sglang" / "test" / "run_eval.py"
        )
        simple_eval_common = types.ModuleType("sglang.test.simple_eval_common")
        simple_eval_common.ChatCompletionSampler = object
        simple_eval_common.CompletionSampler = FakeCompletionSampler
        simple_eval_common.Eval = object
        simple_eval_common.make_report = lambda result: ""
        simple_eval_common.set_ulimit = lambda: None

        spec = importlib.util.spec_from_file_location(
            "_run_eval_under_test", run_eval_path
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "sglang": types.ModuleType("sglang"),
                "sglang.test": types.ModuleType("sglang.test"),
                "sglang.test.simple_eval_common": simple_eval_common,
            },
        ):
            spec.loader.exec_module(module)
        return module

    def _args(self, **overrides):
        args = dict(
            api="completion",
            base_url="http://localhost:30000/v1",
            max_tokens=16,
            model="dummy-model",
            temperature=0.0,
            top_p=1.0,
        )
        args.update(overrides)
        return SimpleNamespace(**args)

    def test_completion_preserves_none_stop(self):
        run_eval_module = self._load_run_eval_module()

        eval_obj = DummyEval()
        run_eval_module.run_eval_once(
            self._args(stop=None),
            "http://localhost:30000/v1",
            eval_obj,
        )

        self.assertEqual(
            FakeCompletionSampler.instances[0].kwargs["stop"],
            None,
        )

    def test_completion_preserves_explicit_stop(self):
        run_eval_module = self._load_run_eval_module()

        eval_obj = DummyEval()
        run_eval_module.run_eval_once(
            self._args(stop=["custom-stop"]),
            "http://localhost:30000/v1",
            eval_obj,
        )

        self.assertEqual(
            FakeCompletionSampler.instances[0].kwargs["stop"],
            ["custom-stop"],
        )


class MusaGemma4SmokeConfigTest(unittest.TestCase):
    def test_gemma4_smoke_has_completion_stop_strings(self):
        test_path = (
            Path(__file__).parents[1]
            / "musa"
            / "test_server_smoke_gemma4_26b_a4b_4gpu.py"
        )
        tree = ast.parse(test_path.read_text())
        default_stop = None
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "default_gsm8k_stop":
                default_stop = tuple(elt.value for elt in node.value.elts)
                break

        self.assertEqual(
            default_stop,
            (
                "Question",
                "Assistant:",
                "<|separator|>",
                "<turn|>",
                "<|tool_response>",
                "<|turn>",
            ),
        )

    def test_only_gemma4_musa_smoke_sets_default_stop_strings(self):
        musa_dir = Path(__file__).parents[1] / "musa"
        files_with_default_stop = []
        for test_path in musa_dir.glob("test_server_smoke_*.py"):
            tree = ast.parse(test_path.read_text())
            if any(
                isinstance(node, ast.keyword) and node.arg == "default_gsm8k_stop"
                for node in ast.walk(tree)
            ):
                files_with_default_stop.append(test_path.name)

        self.assertEqual(
            files_with_default_stop,
            ["test_server_smoke_gemma4_26b_a4b_4gpu.py"],
        )


if __name__ == "__main__":
    unittest.main()

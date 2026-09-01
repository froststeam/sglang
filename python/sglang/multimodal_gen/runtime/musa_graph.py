"""Opt-in MUSA graph replay for fixed-signature diffusion calls."""

from __future__ import annotations

import copy
from contextlib import ExitStack, contextmanager
import logging
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

@contextmanager
def _collective_graph_contexts():
    """Arm custom collective registration once while capturing a DiT graph."""
    try:
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_cfg_group,
            get_sp_group,
            get_tp_group,
        )

        groups = (get_sp_group(), get_tp_group(), get_cfg_group())
    except Exception:
        groups = ()
    with ExitStack() as stack:
        seen = set()
        for group in groups:
            if group is None or id(group) in seen:
                continue
            seen.add(id(group))
            capture = getattr(group, "graph_capture", None)
            if capture is not None:
                stack.enter_context(capture())
        yield


def _clone_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        try:
            cloned = copy.copy(value)
            for key, item in vars(value).items():
                setattr(cloned, key, _clone_tree(item))
            return cloned
        except Exception:
            return value
    return value


def _copy_tree(dst: Any, src: Any) -> None:
    if torch.is_tensor(dst) and torch.is_tensor(src):
        dst.copy_(src)
    elif isinstance(dst, (tuple, list)) and isinstance(src, (tuple, list)):
        for left, right in zip(dst, src):
            _copy_tree(left, right)
    elif isinstance(dst, dict) and isinstance(src, dict):
        for key in dst.keys() & src.keys():
            _copy_tree(dst[key], src[key])
    elif hasattr(dst, "__dict__") and hasattr(src, "__dict__"):
        for key in vars(dst).keys() & vars(src).keys():
            _copy_tree(getattr(dst, key), getattr(src, key))


def _signature(value: Any) -> Any:
    if torch.is_tensor(value):
        return ("tensor", tuple(value.shape), str(value.dtype), value.device.type)
    if isinstance(value, tuple):
        return ("tuple", tuple(_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_signature(item) for item in value))
    if isinstance(value, dict):
        return ("dict", tuple((key, _signature(value[key])) for key in sorted(value)))
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("const", value)
    if hasattr(value, "__dict__"):
        return (
            "object",
            type(value).__module__,
            type(value).__qualname__,
            tuple(sorted(vars(value))),
        )
    return ("object", type(value).__module__, type(value).__qualname__)


class MusaGraphCallable:
    """Lazy fixed-shape graph wrapper; unsupported paths fail open to eager."""

    def __init__(self, fn: Callable[..., Any], *, enabled: bool):
        self.fn = fn
        self.enabled = bool(
            enabled
            and hasattr(torch, "musa")
            and hasattr(torch.musa, "MUSAGraph")
        )
        self._graphs: dict[Any, tuple["_MusaGraphCapture", Any, Any, Any]] = {}
        self._failed = False

    def __call__(self, *args, **kwargs):
        if not self.enabled or self._failed:
            return self.fn(*args, **kwargs)
        capture = None
        try:
            key = (_signature(args), _signature(kwargs))
            entry = self._graphs.get(key)
            if entry is None:
                if len(self._graphs) >= 4:
                    self._failed = True
                    logger.warning(
                        "[MUSA Graph] more than four signatures; using eager fallback"
                    )
                    return self.fn(*args, **kwargs)
                static_args = _clone_tree(args)
                static_kwargs = _clone_tree(kwargs)
                # MUSA requires capture on a non-default stream. Warm up the
                # callable first so lazy kernels are initialized outside capture.
                self.fn(*static_args, **static_kwargs)
                torch.musa.synchronize()
                capture = _MusaGraphCapture()
                output = capture.capture(
                    self.fn, static_args, static_kwargs
                )
                entry = (capture, static_args, static_kwargs, output)
                self._graphs[key] = entry
                logger.info(
                    "[MUSA Graph] captured diffusion transformer call (%d segments)",
                    len(capture.graphs),
                )
            capture, static_args, static_kwargs, output = entry
            return capture.replay(output, args, kwargs)
        except Exception as exc:
            self._failed = True
            if capture is not None:
                capture.reset()
            for capture, *_ in self._graphs.values():
                capture.reset()
            self._graphs.clear()
            logger.warning("[MUSA Graph] fallback to eager: %s", exc)
            return self.fn(*args, **kwargs)


class _MusaGraphCapture:
    """Single-graph runner for a fixed-shape DiT forward."""

    def __init__(self):
        self.graphs: list[Any] = []
        self._static_args = None
        self._static_kwargs = None
        self._current_graph = None
        self.stream = torch.musa.Stream()
        self.pool = torch.musa.graph_pool_handle()

    def _begin(self):
        graph = torch.musa.MUSAGraph()
        graph.capture_begin(pool=self.pool)
        return graph

    def capture(self, fn, args, kwargs):
        self._static_args, self._static_kwargs = args, kwargs
        with _collective_graph_contexts():
            with torch.musa.stream(self.stream):
                graph = self._begin()
                output = fn(*args, **kwargs)
                graph.capture_end()
                self.graphs.append(graph)
        return output

    def replay(self, output, live_args, live_kwargs):
        caller_stream = torch.musa.current_stream()
        self.stream.wait_stream(caller_stream)
        with torch.musa.stream(self.stream):
            _copy_tree(self._static_args, live_args)
            _copy_tree(self._static_kwargs, live_kwargs)
            self.graphs[0].replay()
        caller_stream.wait_stream(self.stream)
        return output

    def reset(self):
        for graph in self.graphs:
            try:
                graph.reset()
            except Exception:
                pass
        self.graphs.clear()

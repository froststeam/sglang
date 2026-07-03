import sys
import types
from typing import Iterable


class _ForwardingModule(types.ModuleType):
    _target_module: types.ModuleType

    def __getattr__(self, name: str):
        return getattr(self._target_module, name)

    def __setattr__(self, name: str, value):
        if name.startswith("__") or name in {"_target_module", "__all__"}:
            return super().__setattr__(name, value)
        setattr(self._target_module, name, value)
        sync_patch = getattr(self._target_module, "_sync_patch_to_domain", None)
        if sync_patch is not None:
            sync_patch(name, value)
        super().__setattr__(name, value)


def install_forwarding_module(module_name: str, target: types.ModuleType, names: Iterable[str]) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        return
    module.__class__ = _ForwardingModule
    module._target_module = target
    module.__all__ = list(names)

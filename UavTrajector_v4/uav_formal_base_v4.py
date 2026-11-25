#!/usr/bin/env python3
"""v4 compatibility entry point for the formal UAV implementation.

The network implementation remains exactly the v2 EMam-SE + MSA-DTP + H-PGD
model.  Version 4 changes only the soft-state training objective and checkpoint
selection in the trainer modules.  Re-exporting every non-dunder name keeps
private helpers such as ``_make_optimizer`` available to the formal trainer.
"""
from __future__ import annotations

import uav_formal_base_v2 as _v2

for _name in dir(_v2):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_v2, _name)

V4_BASE_DESCRIPTION = (
    "v2 architecture with v4 balanced soft-state distillation and "
    "state-aware probabilistic checkpoint selection"
)


def _sync_protocol_overrides_to_v2() -> None:
    """Propagate loader-installed protocol hooks to the implementation module.

    Functions re-exported above retain the v2 module dictionary as their
    global namespace. The formal loader patches this v4 module after import,
    so those hooks must also be copied to the backing module before dataset
    preparation starts.
    """
    for _hook_name in (
        "_protocol_cache_dir",
        "build_window_indices",
        "assign_intent_labels",
    ):
        _hook = globals().get(_hook_name)
        if _hook is not None:
            setattr(_v2, _hook_name, _hook)


def prepare_shared_dataset(config):
    """Synchronise v4 protocol hooks and delegate to the v2 implementation."""
    _sync_protocol_overrides_to_v2()
    return _v2.prepare_shared_dataset(config)

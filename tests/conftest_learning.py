"""
Conftest helper for learning tests.

The main aion.__init__.py imports the full kernel which requires
many heavy dependencies. This module patches the aion package to
allow importing aion.learning.* without triggering the full chain.
"""

import sys
import types


def patch_aion_imports():
    """Replace the aion top-level module with a namespace-only stub."""
    if "aion" in sys.modules:
        return  # Already loaded

    aion_mod = types.ModuleType("aion")
    aion_mod.__path__ = ["/home/user/Aion/aion"]
    aion_mod.__package__ = "aion"
    sys.modules["aion"] = aion_mod

    # Also stub aion.core.kernel so TYPE_CHECKING imports don't fail
    core_mod = types.ModuleType("aion.core")
    core_mod.__path__ = ["/home/user/Aion/aion/core"]
    core_mod.__package__ = "aion.core"
    sys.modules["aion.core"] = core_mod

    kernel_mod = types.ModuleType("aion.core.kernel")
    kernel_mod.__package__ = "aion.core"

    class _StubKernel:
        pass

    kernel_mod.AIONKernel = _StubKernel
    sys.modules["aion.core.kernel"] = kernel_mod


patch_aion_imports()

"""Load the RoboMemArena LIBERO fork before importing the local agent stack."""

from __future__ import annotations

from pathlib import Path
import sys


def activate_robomemarena_core(
    *, checkout_root: str | Path, source_root: str | Path
) -> None:
    """Use RoboMemArena's simulator assets with this repository's agent_env.

    RoboMemArena ships a fork under the same ``libero`` package name.  The
    server is a fresh process, so it can load that fork first and then extend
    the package search path with this repository's ``agent_env`` modules.  The
    Codex process never imports the fork or receives its private raw state.
    """

    checkout = Path(checkout_root).expanduser().resolve()
    source = Path(source_root).resolve()
    fork_root = checkout / "evaluation_benchmark" / "libero_fork"
    if not (fork_root / "libero" / "envs").is_dir():
        raise FileNotFoundError(
            f"RoboMemArena LIBERO fork is unavailable under {fork_root}"
        )
    if "libero" in sys.modules:
        raise RuntimeError(
            "RoboMemArena core must be activated before importing libero"
        )

    sys.path.insert(0, str(fork_root))
    import libero  # pylint: disable=import-outside-toplevel
    import libero.libero as core  # pylint: disable=import-outside-toplevel

    # The upstream checkout contains a compatibility package at
    # libero/libero.  Its benchmark and environment modules expect these path
    # helpers on that compatibility package, while the implementations live on
    # the outer package.
    core.get_libero_path = libero.get_libero_path
    core.get_default_path_dict = libero.get_default_path_dict

    local_outer = str(source / "libero")
    local_inner = str(source / "libero" / "libero")
    if local_outer not in libero.__path__:
        libero.__path__.append(local_outer)
    if local_inner not in core.__path__:
        core.__path__.append(local_inner)

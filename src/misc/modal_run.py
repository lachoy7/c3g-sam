"""Helpers for Modal local entrypoints (blocking .remote vs detached .spawn)."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


def dispatch_remote(
    fn: Callable[..., T],
    /,
    *args: Any,
    detach: bool = False,
    job_name: str = "job",
    app_name: str | None = None,
    **kwargs: Any,
) -> T | Any:
    """
    Run a Modal function or cls method on Modal GPUs.

    When ``detach`` is True, uses ``.spawn()`` and returns immediately with a call id
    (no local GPU, dataset, or blocking wait required). When False, uses ``.remote()``
    and blocks until the job finishes.
    """
    if detach:
        handle = fn.spawn(*args, **kwargs)
        print(f"Detached {job_name} on Modal.")
        print(f"  call_id: {handle.object_id}")
        if app_name:
            print(f"  logs:    modal app logs {app_name}")
        print("  status:  modal app list")
        return handle
    return fn.remote(*args, **kwargs)

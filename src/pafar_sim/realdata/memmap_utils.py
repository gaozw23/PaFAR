"""Centralized, Windows-safe lifecycle management for NumPy memory maps."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import gc
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import numpy as np
import psutil


def _walk_objects(obj: Any, seen: set[int]) -> Iterable[Any]:
    """Yield an object graph, including ndarray ``.base`` ownership chains."""
    if obj is None or id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj
    if isinstance(obj, np.ndarray):
        base = getattr(obj, "base", None)
        if base is not None:
            yield from _walk_objects(base, seen)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _walk_objects(key, seen)
            yield from _walk_objects(value, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for value in obj:
            yield from _walk_objects(value, seen)
    elif is_dataclass(obj) and not isinstance(obj, type):
        for field in fields(obj):
            yield from _walk_objects(getattr(obj, field.name), seen)


def _memmaps_in(obj: Any) -> list[np.memmap]:
    return [value for value in _walk_objects(obj, set()) if isinstance(value, np.memmap)]


def close_memmap(obj: Any) -> None:
    """Close every memmap owning ``obj``; safe for views and repeated calls."""
    for mapping in reversed(_memmaps_in(obj)):
        mmap_handle = getattr(mapping, "_mmap", None)
        if mmap_handle is not None:
            try:
                mmap_handle.close()
            except (BufferError, OSError, ValueError):
                # A previously closed mapping is an idempotent success.  A
                # BufferError is revisited after derived views are collected.
                pass
    gc.collect()


def flush_and_close_memmap(obj: Any) -> None:
    """Flush writable mappings in an object graph and close their OS handles."""
    mappings = _memmaps_in(obj)
    for mapping in mappings:
        try:
            mapping.flush()
        except (BufferError, OSError, ValueError):
            # Closed/read-only mappings can legitimately reject a second flush.
            pass
    close_memmap(obj)


def close_memmap_tree(objects: Any, *, flush: bool = False) -> None:
    """Close mappings nested in containers, dataclasses, and ndarray views."""
    if flush:
        flush_and_close_memmap(objects)
    else:
        close_memmap(objects)


def open_files_under(root: str | Path, *, process: psutil.Process | None = None) -> list[str]:
    """Return files under ``root`` currently open by one process."""
    target = str(Path(root).resolve()).casefold()
    proc = process or psutil.Process()
    try:
        return sorted(
            str(item.path) for item in proc.open_files()
            if str(Path(item.path).resolve()).casefold().startswith(target)
        )
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return []


def assert_no_open_files(root: str | Path, *, process: psutil.Process | None = None) -> None:
    remaining = open_files_under(root, process=process)
    if remaining:
        raise RuntimeError(f"open cache handles remain: {remaining[:8]}")


def bounded_replace(
    source: str | Path,
    target: str | Path,
    *,
    attempts: int = 8,
    initial_delay: float = 0.05,
    replace: Callable[[str | bytes | os.PathLike[str] | os.PathLike[bytes], str | bytes | os.PathLike[str] | os.PathLike[bytes]], None] = os.replace,
) -> list[dict[str, Any]]:
    """Atomically replace one small file with bounded WinError-32 retries."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    source_path, target_path = Path(source), Path(target)
    audit: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        try:
            replace(source_path, target_path)
            return audit
        except OSError as exc:
            retryable = isinstance(exc, PermissionError) or getattr(exc, "winerror", None) == 32
            audit.append({
                "attempt": attempt,
                "error": f"{type(exc).__name__}: {exc}",
                "winerror": getattr(exc, "winerror", None),
                "open_files": open_files_under(source_path.parent),
            })
            if not retryable or attempt == attempts:
                message = f"bounded replace failed after {attempt} attempt(s): {audit}"
                raise PermissionError(message) from exc
            time.sleep(initial_delay * (2 ** (attempt - 1)))
    raise AssertionError("unreachable")

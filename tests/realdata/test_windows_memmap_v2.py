from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pytest

from pafar_sim.realdata import feature_cache as fc
from pafar_sim.realdata.feature_cache import (
    build_feature_cache, commit_completion_marker, compute_cache_id,
)
from pafar_sim.realdata.memmap_utils import (
    bounded_replace, close_memmap, close_memmap_tree, flush_and_close_memmap,
    open_files_under,
)
from pafar_sim.realdata.schema import load_config


ROOT = Path(__file__).resolve().parents[2]


def _worker_write_metadata(path: str) -> dict[str, object]:
    target = Path(path)
    mapping = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(32, 4))
    mapping[:] = 3.0
    flush_and_close_memmap(mapping)
    return {"path": str(target), "rows": 32, "contains_memmap": False, "pid": os.getpid()}


def test_memmap_flush_close_and_replace(tmp_path: Path):
    source, target = tmp_path / "source.npy", tmp_path / "target.npy"
    mapping = np.lib.format.open_memmap(source, mode="w+", dtype=np.float32, shape=(8, 2))
    mapping[:] = 7
    flush_and_close_memmap(mapping)
    os.replace(source, target)
    assert target.is_file() and not open_files_under(tmp_path)


def test_memmap_view_releases_base(tmp_path: Path):
    source, target = tmp_path / "view.npy", tmp_path / "renamed.npy"
    mapping = np.lib.format.open_memmap(source, mode="w+", dtype=np.float32, shape=(16, 3))
    outer = mapping[2:12][:, 1:]
    assert isinstance(outer.base, np.ndarray)
    close_memmap_tree(outer, flush=True)
    os.replace(source, target)
    assert target.exists()


def test_loaded_mmap_is_closed(tmp_path: Path):
    source, target = tmp_path / "loaded.npy", tmp_path / "loaded-renamed.npy"
    np.save(source, np.arange(20, dtype=np.int32), allow_pickle=False)
    loaded = np.load(source, mmap_mode="r", allow_pickle=False)
    assert isinstance(loaded, np.memmap)
    close_memmap(loaded)
    os.replace(source, target)
    assert target.exists()


def test_close_memmap_is_idempotent(tmp_path: Path):
    path = tmp_path / "twice.npy"
    mapping = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(4,))
    close_memmap(mapping)
    close_memmap(mapping)


def test_worker_does_not_return_memmap(tmp_path: Path):
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        result = pool.submit(_worker_write_metadata, str(tmp_path / "worker.npy")).result()
    assert result["contains_memmap"] is False
    assert not any(isinstance(value, np.memmap) for value in result.values())


def test_pool_shutdown_before_finalize(tmp_path: Path):
    parent = psutil.Process()
    before = {child.pid for child in parent.children(recursive=True)}
    pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
    result = pool.submit(_worker_write_metadata, str(tmp_path / "pool.npy")).result()
    pool.shutdown(wait=True)
    after = {child.pid for child in parent.children(recursive=True) if child.is_running()}
    assert not (after - before)
    commit_completion_marker(tmp_path, {"status": "complete", "worker": result}, validated=True)


def test_incomplete_cache_rejected(tmp_path: Path):
    (tmp_path / "CACHE_BUILDING.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete"):
        fc._load_marker(tmp_path)


def test_complete_marker_written_last(tmp_path: Path):
    with pytest.raises(RuntimeError, match="before validation"):
        commit_completion_marker(tmp_path, {"status": "complete"}, validated=False)
    assert not (tmp_path / "CACHE_COMPLETE.json").exists()
    commit_completion_marker(tmp_path, {"status": "complete"}, validated=True)
    assert (tmp_path / "CACHE_COMPLETE.json").exists()


def test_cache_id_changes_with_source():
    common = dict(raw_manifest_checksum="r", cohort_checksum="c", split_checksum="s",
                  feature_checksum="f", config_checksum="g", master_seed=20260804)
    first = compute_cache_id(source_checksum="source-1", **common)
    second = compute_cache_id(source_checksum="source-2", **common)
    assert first != second


def test_fresh_cache_does_not_reuse_failed_partial(tmp_path: Path):
    config = load_config(ROOT / "configs" / "realdata_primary.yaml")
    root = tmp_path / "realdata_features_failed"
    root.mkdir(); (root / "CACHE_BUILDING.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to reuse"):
        build_feature_cache(config, pd.DataFrame(), cache_id="failed", identity={},
                            n_jobs=1, fresh=False, cache_root_override=root, write_outputs=False)


def test_bounded_replace_retry(tmp_path: Path):
    source, target = tmp_path / "small.tmp", tmp_path / "small.json"
    source.write_text("ok", encoding="utf-8")
    calls = []
    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            error = PermissionError("simulated WinError 32")
            error.winerror = 32
            raise error
        os.replace(src, dst)
    audit = bounded_replace(source, target, attempts=4, initial_delay=0, replace=flaky)
    assert len(audit) == 2 and len(calls) == 3 and target.exists()
    source.write_text("again", encoding="utf-8")
    with pytest.raises(PermissionError, match="after 2 attempt"):
        bounded_replace(source, target, attempts=2, initial_delay=0,
                        replace=lambda *_: (_ for _ in ()).throw(PermissionError("busy")))


def test_no_open_cache_files_before_complete_marker(tmp_path: Path):
    mapping = np.lib.format.open_memmap(tmp_path / "open.npy", mode="w+", dtype=np.float32, shape=(4,))
    assert open_files_under(tmp_path)
    with pytest.raises(RuntimeError, match="open cache handles"):
        commit_completion_marker(tmp_path, {"status": "complete"}, validated=True)
    flush_and_close_memmap(mapping)
    commit_completion_marker(tmp_path, {"status": "complete"}, validated=True)


def test_resume_complete_cache_only_validates_checksums(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = load_config(ROOT / "configs" / "realdata_primary.yaml")
    root = tmp_path / "realdata_features_cache"
    root.mkdir()
    (root / "CACHE_COMPLETE.json").write_text('{"status":"complete","cache_id":"cache"}', encoding="utf-8")
    monkeypatch.setattr(fc, "_validate_cache", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("array checksum mismatch")))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        build_feature_cache(config, pd.DataFrame(), cache_id="cache", identity={},
                            n_jobs=1, fresh=False, cache_root_override=root, write_outputs=False)

# Windows Memmap Failure Analysis

## Incident

The first formal real-data run stopped before learner fitting at
`src/pafar_sim/realdata/feature_cache.py:107` in the failed implementation:

```python
for array, temp, target in arrays:
    array.flush()
    del array
    os.replace(temp, target)
```

The first failing operation attempted to rename:

- source: `data/physionet2019/cache/features_A.values.npy.9wmp4o1c.tmp.npy`
- target: `data/physionet2019/cache/features_A.values.npy`
- exception: `PermissionError: [WinError 32]`

## Exact ownership path

The failed builder created seven `numpy.memmap` objects using
`numpy.lib.format.open_memmap`. It retained each object in the `arrays` list and
also unpacked the same objects into `x`, `patient_codes`, `hours`, `labels`,
`onset`, `horizon`, and `event`. Inside the finalization loop, `del array`
deleted only the loop variable. It did not remove either the `arrays` tuple
reference or the seven unpacked references. Therefore the main Python process
still owned the `_mmap` for `features_A.values...tmp.npy` when `os.replace` ran.

The failure call chain was:

```text
raw patient PSV
  -> build_patient_features
  -> assignment into hospital-level open_memmap arrays
  -> memmap.flush
  -> del loop variable only (owners remain)
  -> os.replace on large temporary values array
  -> WinError 32
```

## Required incident questions

1. **Which rename failed?** The A-hospital values array listed above.
2. **Which memmap remained open?** The `numpy.memmap` stored both in
   `arrays[0][0]` and `x`; its `_mmap` owned the source-file handle.
3. **Did a derived ndarray view retain `.base`?** No persistent derived view was
   found on this traceback path. Temporary assignment slices existed only for
   individual statements. The direct memmap references alone prove the cause.
   V2 nevertheless recursively closes ndarray `.base` chains to prevent this
   related Windows failure mode.
4. **Was an `np.load(..., mmap_mode=...)` object left open?** No. The exception
   occurred during cache construction before the read-only cache loader ran.
5. **Did a joblib/multiprocessing worker remain alive?** No. V1 cache writing was
   single-process and did not use joblib or multiprocessing.
6. **Was the main process the owner?** Yes. The direct Python references were in
   the formal main process. After that process exited, psutil found zero Python
   processes with files open under the failed cache root.
7. **Could a logger, checksum reader, validator, antivirus, or indexer be the
   owner?** No logger, checksum reader, or validator had opened the array before
   the failing rename. An external antivirus/indexer cannot be proven absent
   historically, but it is unnecessary to explain the deterministic direct
   handle retained by the main process. Post-failure psutil auditing found no
   remaining cache handles.
8. **Where did WinError 32 first arise?** The first and only traceback source was
   failed `feature_cache.py:107`, at `os.replace(temp, target)`.

## V2 engineering correction

V2 adds `memmap_utils.py` as the only implementation allowed to access a
memmap's `_mmap`. It flushes and closes direct maps, recursively follows ndarray
`.base` chains, traverses containers/dataclasses, is idempotent, performs
`gc.collect()`, and audits current-process files with psutil.

The cache design is now immutable and versioned:

```text
data/physionet2019/cache/realdata_features_<cache_id>/
  arrays/
  blocks/
  metadata/
  CACHE_BUILDING.json
  CACHE_COMPLETE.json
```

Large arrays are written directly to their unique final version directory and
are never renamed. Independent hospital workers write disjoint files, close all
maps, and return only JSON-compatible metadata. The main process waits for the
pool to shut down, checks for child-process and open-file leaks, validates array
and block checksums, checks patient/time uniqueness and feature causality, and
compares reference features. Only then does it atomically replace a small
temporary JSON file with `CACHE_COMPLETE.json`. The replace helper has at most
eight exponentially backed-off attempts and fails clearly when exhausted.

Any directory containing only `CACHE_BUILDING.json` is incomplete. It is never
accepted by `--resume` and cannot reach learner fitting.

## Evidence and tests

The V1 partial tree, file metadata, SHA-256 hashes, representative NPY header,
failed lock, logs, traceback, package versions, and platform details are stored
under `outputs/realdata/failure_archive/winerror32_20260804T120337/`. The large
partial arrays were not duplicated into the archive.

Thirteen Windows regression tests cover flush/close/replace, ndarray views,
loaded read-only maps, idempotence, worker return types, pool shutdown,
incomplete-cache rejection, marker ordering, cache-ID source sensitivity,
failed-partial isolation, bounded retry, no-open-file marker gating, and
checksum validation on resume.

The independent Windows stress gate completed 26/26 cycles with zero WinError
32, zero open-handle failures, and zero active child processes after shutdown.

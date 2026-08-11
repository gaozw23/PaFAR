"""Order-independent random-number stream management."""
from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np


def _encode_part(part: Any) -> list[int]:
    digest = sha256(str(part).encode("utf-8")).digest()
    return list(np.frombuffer(digest[:16], dtype=np.uint32).astype(int))


def child_seed(master_seed: int, *parts: Any) -> int:
    """Derive a stable uint64 seed without Python's process-randomized hash."""
    entropy: list[int] = [int(master_seed) & 0xFFFFFFFF]
    for part in parts:
        entropy.extend(_encode_part(part))
    state = np.random.SeedSequence(entropy).generate_state(2, dtype=np.uint32)
    return int(state[0]) | (int(state[1]) << 32)


def make_rng(master_seed: int, *parts: Any) -> np.random.Generator:
    """Create a PCG64DXSM generator, falling back to PCG64 when unavailable."""
    seed = child_seed(master_seed, *parts)
    bitgen_type = getattr(np.random, "PCG64DXSM", np.random.PCG64)
    return np.random.Generator(bitgen_type(seed))


def bit_generator_name() -> str:
    """Name the selected bit generator for manifests."""
    return "PCG64DXSM" if hasattr(np.random, "PCG64DXSM") else "PCG64"


def seed_record(master_seed: int, experiment: str, scenario: str, replicate: int, condition: str = "primary") -> dict[str, int]:
    """Return all prespecified independent child streams for a replicate."""
    roles = (
        "replicate", "validation", "calibration", "source_site",
        "target_reservoir", "target_reservoir_order", "test_non_events", "test_events", "xgboost",
        "oracle_reference", "training",
    )
    return {
        role: child_seed(master_seed, experiment, scenario, condition, replicate, role)
        for role in roles
    }

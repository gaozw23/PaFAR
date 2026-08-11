import json
import pytest

from pafar_sim.exp1.oracle import build_oracle, load_oracle, oracle_filename


def test_oracle_filename_and_metadata_prevent_stale_reuse(tmp_path):
    kwargs = dict(hmax=12, tmin=6, smooth_length=3)
    name = oracle_filename("S1", "A", 30, 7, **kwargs)
    assert all(token in name for token in ("S1_A", "t6", "h12", "L3", "seed7", "N30"))
    path = tmp_path / name
    built = build_oracle(path, "S1", "A", 30, 7, chunk_size=10, **kwargs)
    loaded = load_oracle(path, "S1", "A", 30, 7, **kwargs)
    assert (built == loaded).all()
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["smooth_length"] = 2
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Stale/incompatible"):
        load_oracle(path, "S1", "A", 30, 7, **kwargs)


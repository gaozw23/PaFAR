import numpy as np

from pafar_sim.calibration import initial_boundaries, merge_sparse_bins, fit_time_template


def test_primary_bins():
    assert initial_boundaries(6, 120) == (6, 12, 24, 48, 72, 120)


def test_latest_sparse_bin_is_merged_to_predecessor():
    eligible = np.zeros((100, 120), dtype=bool)
    eligible[:60, 5:12] = True
    eligible[:60, 12:24] = True
    eligible[:40, 24:48] = True
    eligible[:70, 48:72] = True
    eligible[:30, 72:120] = True
    boundaries, counts = merge_sparse_bins(eligible, initial_boundaries(6, 120), 50)
    # Latest sparse final bin merges left first; subsequent recomputation leaves valid bins.
    assert boundaries[-1] == 120
    assert all(c >= 50 for c in counts) or len(counts) == 1


def test_empty_single_bin_template_convention():
    score = np.zeros((3, 20))
    eligible = np.zeros_like(score, dtype=bool)
    template = fit_time_template(score, eligible, 6, 20)
    assert template.locations == (0.0,) and template.scales == (1.0,)


from pafar_sim.config import apply_condition, effective_config_checksum, load_config


def test_one_factor_conditions_are_separate():
    loaded = load_config("configs/exp1_sensitivity.yaml")
    calibration = apply_condition(loaded.data, "calibration_size")
    weak = apply_condition(loaded.data, "weak_signal")
    assert calibration["experiment1"]["calibration_sizes"] == [100, 250, 1000]
    assert calibration["experiment1"]["signal"] == 1.5
    assert weak["experiment1"]["signal"] == 1.0 and weak["alpha"] == .10


def test_effective_checksum_changes_with_condition_and_master_seed():
    loaded = load_config("configs/exp1_sensitivity.yaml")
    primary = apply_condition(loaded.data, None)
    weak = apply_condition(loaded.data, "weak_signal")
    reseeded = apply_condition(loaded.data, None); reseeded["master_seed"] += 1
    assert len({effective_config_checksum(primary), effective_config_checksum(weak), effective_config_checksum(reseeded)}) == 3

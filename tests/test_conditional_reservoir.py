import numpy as np

import pafar_sim.exp2.dgp as dgp


def _assert_batch_equal(left, right):
    for field in left.__dataclass_fields__:
        np.testing.assert_allclose(getattr(left, field), getattr(right, field), equal_nan=True)


def test_conditional_non_event_reservoir_is_natural_reproducible_and_nested(monkeypatch):
    original = dgp.generate_exp2
    force_values = []
    def observed_generate(*args, **kwargs):
        force_values.append(kwargs.get("force_event"))
        return original(*args, **kwargs)
    monkeypatch.setattr(dgp, "generate_exp2", observed_generate)
    kwargs = dict(n=300, scenario="E3", site="B", hmax=20, tmin=6)
    first, info1 = dgp.generate_exp2_non_events(
        np.random.default_rng(2), order_rng=np.random.default_rng(3), **kwargs,
    )
    second, info2 = dgp.generate_exp2_non_events(
        np.random.default_rng(2), order_rng=np.random.default_rng(3), **kwargs,
    )
    assert force_values and set(force_values) == {None}
    assert (~first.event).all() and np.isposinf(first.onset).all()
    assert np.array_equal(first.patient_ids, np.arange(300))
    assert info1 == info2 and info1.candidates >= 300 and 0 < info1.acceptance_rate < 1
    _assert_batch_equal(first, second)
    np.testing.assert_array_equal(first.patient_ids[:100], np.arange(100))
    np.testing.assert_array_equal(first.patient_ids[:250], np.arange(250))


def test_conditional_baselines_match_independent_natural_d0_and_shift_from_unconditional():
    reservoir, _ = dgp.generate_exp2_non_events(
        np.random.default_rng(20), 5000, "E3", order_rng=np.random.default_rng(21),
        site="B", hmax=12, tmin=6,
    )
    natural = dgp.generate_exp2(np.random.default_rng(22), 9000, "E3", site="B", hmax=12, tmin=6)
    non = ~natural.event
    for a, b in (
        (reservoir.age_covariate, natural.age_covariate[non]),
        (reservoir.binary_covariate, natural.binary_covariate[non]),
        (reservoir.random_effect, natural.random_effect[non]),
    ):
        assert abs(a.mean() - b.mean()) < .06
    reservoir_lp = .30 * reservoir.age_covariate + .20 * reservoir.binary_covariate + .35 * reservoir.random_effect
    unconditional_lp = .30 * natural.age_covariate + .20 * natural.binary_covariate + .35 * natural.random_effect
    assert reservoir_lp.mean() < unconditional_lp.mean() - .02


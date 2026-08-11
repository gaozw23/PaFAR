import numpy as np

from pafar_sim.exp2.learner import patient_average_positive_fraction, training_weights, validation_weights


def test_training_and_validation_weights():
    labels = np.array([0, 1, 1, 0, 0])
    patient = np.array([0, 0, 1, 1, 1])
    assert patient_average_positive_fraction(labels, patient) == np.mean([.5, 1/3])
    weights, _ = training_weights(labels, patient)
    assert weights.mean() == 1
    val = validation_weights(patient)
    assert val[patient == 0].sum() == val[patient == 1].sum()


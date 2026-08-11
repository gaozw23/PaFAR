import numpy as np
from pafar_sim.exp2.learner import training_weights, validation_weights
def test_patient_equal_class_balanced_training_weights():
    y=np.array([0,0,1,0,1]); p=np.array([0,0,0,1,1]); w,pi=training_weights(y,p); assert 0<pi<1 and np.isclose(w.mean(),1)
def test_validation_patient_totals_equal():
    p=np.array([0,0,0,1,1]); w=validation_weights(p); assert np.isclose(w[p==0].sum(),w[p==1].sum())


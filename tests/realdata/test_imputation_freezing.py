import numpy as np
from pafar_sim.realdata.imputation import fit_preprocessor
def test_training_only_medians_and_target_freeze():
    x=np.array([[1,np.nan],[3,2],[999,999]],dtype=np.float32); mask=np.array([1,1,0],bool); p=fit_preprocessor([x],[mask],("x_last","z_mean_6h")); assert np.array_equal(p.medians,[2,2]); before=p.checksum; x[2]=0; assert fit_preprocessor([x],[mask],("x_last","z_mean_6h")).checksum==before; assert np.isfinite(p.transform(x)).all()


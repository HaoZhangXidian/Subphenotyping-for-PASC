"""
===========================================
Train topic model
===========================================

"""

# Author: Hao Zhang
# License: Apache License Version 2.0


import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix
from pydpm.model import PFA

data = sio.loadmat('../dataset/COVID_positive_any_new_PASC_Feature_DX.mat')
X = np.array(data['Feature_DX'].T, order='C')

K=10

model=PFA(K, 'cpu')
model.initial(X)
model.train(iter_all=1000, burn_in = 500)
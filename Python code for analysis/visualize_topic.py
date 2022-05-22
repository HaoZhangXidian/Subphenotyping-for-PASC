"""
===========================================
Train topic model
===========================================

"""

# Author: Hao Zhang
# License: Apache License Version 2.0


import numpy as np
import scipy.io as sio


data = sio.loadmat('../trained_topic_model/PFA_trained_model.mat')
Topic = data['Phi_mean']

# For visualization of topic, considering the data privacy, here we just simulate the dictionary of 137 PASC.
# For real data, you should put real PASC name dictionary here.
Dic = []
for i in range(137):
    Dic.append('pasc'+str(i))

K = Topic.shape[1]
top_k = 10

for i in range(K):
    print('Topic '+str(i)+':')
    topic = Topic[:, i]
    topic_ordered_index = np.argsort(topic)[::-1]

    name = ''
    for j in range(top_k):
        name += Dic[topic_ordered_index[j]]+' '
    print(name)


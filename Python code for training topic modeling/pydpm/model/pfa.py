"""
===========================================
Poisson Factor Analysis
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import numpy as np
from ..utils import Model_Sampler_CPU
import time
from ..utils.Metric import *
import scipy.io as sio

realmin = 2.2e-10

class PFA(object):

    def __init__(self, K, device='cpu'):

        self.K = K

        if device == 'cpu':
            self.device = 'cpu'
            self.Multrnd_Matrix = Model_Sampler_CPU.Multrnd_Matrix
            self.Crt_Matrix = Model_Sampler_CPU.Crt_Matrix
            self.Crt_Multirnd_Matrix = Model_Sampler_CPU.Crt_Multirnd_Matrix

        elif device == 'gpu':
            self.device = 'gpu'
            from pydpm.utils import Model_Sampler_GPU
            self.Multrnd_Matrix = Model_Sampler_GPU.Multrnd_Matrix_GPU
            self.Crt_Matrix = Model_Sampler_GPU.Crt_Matrix_GPU
            self.Crt_Multirnd_Matrix = Model_Sampler_GPU.Crt_Multirnd_Matrix_GPU

    def initial(self, data):

        self.data = data
        self.V = data.shape[0]
        self.N = data.shape[1]

        Supara = {}
        Supara['a0pj'] = 0.01
        Supara['b0pj'] = 0.01
        Supara['e0cj'] = 1
        Supara['f0cj'] = 1
        # Supara['eta'] = np.ones(self.T) * 0.01
        self.Eta = 0.01
        Eta = []
        self.Phi = 0.2 + 0.8 * np.random.rand(self.V, self.K)
        self.Phi = self.Phi / np.maximum(realmin, self.Phi.sum(0))

        r_k = np.ones([self.K, 1])/self.K

        self.Theta = np.ones([self.K, self.N]) / self.K
        c_j = np.ones([1, self.N])
        p_j = np.ones([1, self.N])

        self.Supara = Supara
        self.r_k = r_k
        self.c_j = c_j
        self.p_j = p_j


    def train(self, iter_all=200, burn_in=100, step=1):

        data = self.data
        self.Likelihood = []

        Theta_ave = 0
        Phi_ave = 0
        flag = 0


        for iter in range(iter_all):

            start_time = time.time()

            # ======================== Upward Pass ======================== #
            # Update Phi
            Xt_to_t1, WSZS = self.Multrnd_Matrix(data, self.Phi, self.Theta)

            self.Phi = self.Update_Phi(WSZS, self.Eta)

            # ======================== Downward Pass ======================== #

            shape = np.repeat(self.r_k, self.N, axis=1)
            self.Theta = self.Update_Theta(Xt_to_t1, shape)

            likelihood = Poisson_Likelihood(data, np.dot(self.Phi, self.Theta)) / self.V

            if (iter > burn_in) & (np.mod(iter, step) == 0):
                Theta_ave = Theta_ave + self.Theta
                Phi_ave = Phi_ave + self.Phi
                flag = flag + 1
                Theta_mean = Theta_ave / flag
                Phi_mean = Phi_ave / flag
                likelihood_mean = Poisson_Likelihood(data, np.dot(Phi_mean, Theta_mean)) / self.V

            end_time = time.time()


            if (iter > burn_in) & (np.mod(iter, step) == 0):
                print("Epoch {:3d} takes {:8.2f} seconds; Likelihood {:8.2f}, Likelihood mean {:8.2f}".format(iter, end_time-start_time, likelihood, likelihood_mean))
            else:
                print("Epoch {:3d} takes {:8.2f} seconds; Likelihood {:8.2f}".format(iter, end_time - start_time, likelihood))

        sio.savemat('../trained_topic_model/PFA_trained_model.mat',{'Phi_mean': Phi_mean, 'Theta_mean': Theta_mean})




    def Calculate_pj(self, c_j, T):

        # calculate p_j from layer 1 to T+1
        p_j = []
        N = c_j[1].size
        p_j.append((1 - np.exp(-1)) * np.ones([1, N]))  # p_j_1
        p_j.append(1 / (1 + c_j[1]))                    # p_j_2

        for t in [i for i in range(T + 1) if i > 1]:    # p_j_3_T+1; only T>=2 works
            tmp = -np.log(np.maximum(1 - p_j[t - 1], realmin))
            p_j.append(tmp / (tmp + c_j[t]))

        return p_j

    def Update_Phi(self, WSZS_t, Eta_t):

        # update Phi_t
        Phi_t_shape = WSZS_t + Eta_t
        Phi_t = np.random.gamma(Phi_t_shape, 1)
        Phi_t = Phi_t / Phi_t.sum(0)

        return Phi_t

    def Update_Theta(self, Xt_to_t1_t, shape):

        # update Theta_t
        Theta_t_shape = Xt_to_t1_t + shape
        Theta_t = np.random.gamma(Theta_t_shape, 1)

        return Theta_t


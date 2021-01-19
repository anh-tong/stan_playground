import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


EPS = 1e-6

class IBP(nn.Module):

    def __init__(self, n_objects, n_features, alpha=1., temperature=0.01):

        super().__init__()
        self.n_objects = n_objects
        self.n_features = n_features
        self.alpha = alpha
        self.temperature = temperature

        self.initialize_params()


    def initialize_params(self):

        # Benoulli
        nu = np.random.randn(self.n_objects, self.n_features)
        self.nu = nn.Parameter(torch.tensor(nu))

        # Beta
        alpha_K = self.alpha / self.n_features
        q_tau = np.ones((2, self.n_features))
        q_tau[0,:] = alpha_K * np.ones((1, self.n_features))
        q_tau = q_tau + 1.5 * min(1, alpha_K) * (np.random.rand(2, self.n_features) - 1.5)

        self.q_tau_ = nn.Parameter(torch.tensor(q_tau))
        self.q_tau_.requires_grad = False # this one is not trainable

    @property
    def probs(self):
        return 0.5 * (1. + torch.sigmoid(self.nu))

    @property
    def q_tau(self):
        return self.q_tau_


    def compute_expect_pi(self):


        sum_tau = torch.sum(self.q_tau, dim=0)
        tau_1 = torch.reshape(self.q_tau[0,:], sum_tau.shape)
        digamma_sum = torch.digamma(tau_1) - torch.digamma(sum_tau)
        digamma_sum = digamma_sum.sum()
        expect_pi = self.n_features * np.log(self.alpha / self.n_features) + (self.alpha/ self.n_features - 1.) * digamma_sum

        # this one does not involve optimization at all
        return expect_pi

    def compute_expect_z(self):

        psi_1 = torch.digamma(torch.reshape(self.q_tau[0], [self.n_features, 1]))
        psi_2 = torch.digamma(torch.reshape(self.q_tau[1], [self.n_features, 1]))

        probs = self.probs

        nu_psi_1 = psi_1.t() @ probs.t()
        nu_psi_2 = psi_2.t() @ (1. - probs).t()

        # the second term does not contribute to gradient computation
        return (nu_psi_1 + nu_psi_2).sum() +\
               self.n_objects * torch.digamma(self.q_tau[0,:] + self.q_tau[1,:]).sum()

    def compute_entropy(self):

        probs = self.probs
        log_probs = torch.log(probs + EPS)
        c_probs = 1. - probs
        log_c_probs = torch.log(c_probs + EPS)

        entropy = -probs.multiply(log_probs) - c_probs.multiply(log_c_probs)
        return entropy.sum()


    def kl_divergence(self):

        return -self.compute_expect_pi() - self.compute_expect_z() - self.compute_entropy()

    def forward(self):
        """Sample Gumbel Softmaxs"""
        probs = self.probs
        c_probs = 1. - probs
        logits = torch.stack([probs, c_probs], dim=-1).log()
        sample = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        return sample[...,0]


    def update_tau(self):
        sum_prob = self.probs.sum(dim=0)
        tau_1 = self.alpha / self.n_features + sum_prob
        tau_2 = 1. + torch.sum(1 - self.probs, dim=0)
        new_tau = torch.stack([tau_1, tau_2], dim=0)
        with torch.no_grad():
            self.q_tau.copy_(new_tau)

class IndependentGaussian(nn.Module):

    def __init__(self, K):
        super().__init__()
        self.K = K
        self.standard_gaussian = torch.distributions.Normal(loc=torch.zeros((self.K, 1)),
                                                            scale=torch.ones((self.K, 1)))
        self.init_params()

    def init_params(self):
        self.mean = nn.Parameter(torch.randn((self.K, 1)))
        self.log_var = nn.Parameter(torch.zeros((self.K, 1))*np.log(0.01))

        self.gaussian = torch.distributions.Normal(loc=self.mean,
                                                   scale=0.5* self.log_var.exp())

    def forward(self):
        return self.gaussian.rsample()

    def kl_divergence(self):
        return torch.distributions.kl_divergence(self.gaussian, self.standard_gaussian).sum()


class LinearModel(nn.Module):

    def __init__(self, N, D, K):
        super().__init__()
        self.N, self.D, self.K = N, D, K
        self.Z_module = IBP(N, K)
        self.A_module = IBP(D, K)
        self.c_module = IndependentGaussian(K)


    def forward(self):
        Z = self.Z_module()
        A = self.A_module()
        c = self.c_module()
        diag = torch.diag(c.squeeze()).type(torch.DoubleTensor)
        At = A.t()
        return Z @ (diag @ At)

    def likelihood(self, X, sigma2):
        X_hat = self.forward()
        return -0.5 * torch.square(X_hat - X).sum() / (self.N * self.D * sigma2) # - self.kl_divergence()

    def kl_divergence(self):
        kl = self.Z_module.kl_divergence() + self.A_module.kl_divergence() + self.c_module.kl_divergence()
        return kl

    def update_tau(self):
        self.Z_module.update_tau()
        self.A_module.update_tau()





# test
N, D, K = 4, 4, 4

X = torch.randn(N, D)

sigma2 = 0.01

model = LinearModel(N=N, D=D, K=K)

optimizer = torch.optim.Adam(params=list(model.parameters()), lr=0.01)

for i in range(10000):
    # optimizer.zero_grad()
    loss = - model.likelihood(X, sigma2)
    loss.backward(retain_graph=True)
    optimizer.step()
    print(f"Iter {i} \t Loss: {loss.item()}")










import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class IBP(nn.Module):

    def __init__(self, n_objects, n_dims, n_features, alpha, temperature=0.01):

        super().__init__()
        self.n_objects = n_objects
        self.n_dims = n_dims
        self.n_features = n_features
        self.alpha = alpha
        self.temperature = temperature

        self.initialize_params()


    def initialize_params(self):
        tau = np.random.randn(self.n_objects, self.n_features)
        self.tau = nn.Parameter(torch.tensor(tau))

    @property
    def probs(self):
        return 0.5 * (1. + torch.sigmoid(self.tau))

    def forward(self):
        """Sample Gumbel Softmaxs"""
        probs = self.probs
        c_probs = 1 - probs
        logits = torch.stack([probs, c_probs], dim=-1).log()
        sample = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        return sample[...,0]



ibp = IBP(2,2,3, 1.)


for i in range(3):
    print(ibp())






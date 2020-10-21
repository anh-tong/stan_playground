import pystan
import numpy as np

import matplotlib.pyplot as plt

stan_code = """
data {
    int<lower=1> N;
    int<lower=1> D; 
    int<lower=1> num_feature;
    matrix[N, D] x;
    matrix[D, num_feature] w;
    
}

transformed data {
    //
    real scale;
    scale = sqrt(1./N);
}

parameters {
    vector[num_feature] eta;
}

model {
    eta ~ std_normal();
}

generated quantities {
    vector[N] y;
    matrix[N, num_feature] phi_x;
    phi_x = cos(x*w) * scale;
    y = phi_x * eta;
}
"""

N = 100
D = 1
num_feature = 100
x = np.linspace(0, 10, N)[:,None]
w = np.random.randn(D, num_feature)

data = dict(N=N, D=D, num_feature=num_feature, x=x, w=w)
model = pystan.StanModel(model_code=stan_code)
fit = model.sampling(data=data, iter=100)

output = fit.extract(permuted=True)
y = output["y"]

for i in range(10):
    plt.plot(x[:], y[i,:])
plt.show()
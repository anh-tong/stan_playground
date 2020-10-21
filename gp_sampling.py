import pystan

"""
Implement GP sampling in the paper
https://arxiv.org/pdf/2002.09309.pdf

For implementation of GP in Stan, the reading is from
https://mc-stan.org/events/stancon2017-notebooks/stancon2017-trangucci-hierarchical-gps.pdf

Outline:
    1. Random fourier feature of the kernel
    2. Inducing point: implement sampling technique?
"""

stan_code = """

data{
    int<lower=1> N;  // number of data
    int<lower=1> N_pred; // number of test
    int<lower=1> M; // number of inducing points
    vector[N] y;
    real x[N];
    real x_pred[N_pred]; 
}

parameters {
    //kernel parameters
    real<lower=0> length_scale;
    real<lower=0> alpha;
    real<lower=0> sigma;
    //inducing point location
    real z[M];
    vector[M] eta;
}

transformed parameters {
    real jitter = 1e-12;
    
    vector[M] u;
    {
        matrix[M, M] K_uu;
        matrix[M, M] L_uu;
        matrix[N, M] K_fu;
        K_uu = cov_exp_quad(z, alpha, length_scale);
        K_fu = cov_exp_quad(x, z, alpha, length_scale);
        for (m in 1:M)
            K_uu[m,m] = K_uu[m,m] + jitter;
        L_uu = cholesky_decompose(K_uu);
        u = L_uu * eta;
        
    }
}

model {
    
}

"""

N = 3,
N_pred = 2,
M = 1
x = [0., 0.1, 0.2]
x_pred = [0.05, 0.15]
y = [1.1, 0.6, 1.4]

data = {'N': N,
        'N_pred': N_pred,
        'M': M,
        'x': x,
        'y': y,
        'x_pred': x_pred}

model = pystan.StanModel(model_code=stan_code)

model.sampling(data=data, iter=1000, chains=4)

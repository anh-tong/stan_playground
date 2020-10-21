import pystan
import numpy as np

import matplotlib.pyplot as plt

from utils import create_model


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


functions {
        matrix rff(real[] x, matrix omega, real length_scale, real alpha){
                int D = rows(omega);
                int num_feature = cols(omega);
                int N = size(x);
                matrix[N, num_feature] phi_x;
                real scale = sqrt(0.5/N);
                phi_x = cos(to_matrix(x, N, D) * omega / length_scale);
                return alpha * phi_x * scale;
        }
        
        vector cholesky_solve(matrix Luu, vector y){
                int M = rows(Luu);
                vector[M] ret;
                ret = mdivide_left_tri_low(Luu, y);
                ret = mdivide_right_tri_low(ret', Luu)';
                return ret;
        }
}

data{
    int<lower=1> N;  // number of data
    int<lower=1> N_pred; // number of test
    int<lower=1> M; // number of inducing points
    int<lower=1> D; // number of dimensions
    vector[N] y;
    real x[N];
    real x_pred[N_pred]; 
    
    //inducing point. NOTE: fixed inducing location
    real z[M];
    vector[M] u;
    
    //kernel parameters
    real<lower=0> length_scale;
    real<lower=0> alpha;
    real<lower=0> sigma;
    
    // random fourier feature
    int<lower=1> num_feature;
    matrix[D,num_feature] omega;
}

parameters {
    
    //inducing point location
    vector[M] eta;
    
    vector[num_feature] eta_omega;
}

transformed parameters {
    real jitter = 1e-6;
    
    matrix[M, M] Kuu;
    matrix[M, M] Luu;
    matrix[N, M] Kfu;
    matrix[N, N] Kff;
    matrix[N, num_feature] phi_x;
    matrix[M, num_feature] phi_z;
    vector[M] Kuu_div_u;
    matrix[M, N] var_pred;
    //vector[M] u;
    
    
    Kuu = cov_exp_quad(z, alpha, length_scale);
    Kfu = cov_exp_quad(x, z, alpha, length_scale);
    
    for (m in 1:M)
        Kuu[m,m] = Kuu[m,m] + jitter;
    Luu = cholesky_decompose(Kuu);
    // u = Luu * eta;
    
    
    phi_x = rff(x, omega, length_scale, alpha);
    phi_z = rff(z, omega, length_scale, alpha);
    
    //vector[N] prior = phi_x * eta_omega;
    
    //vector[N] update = Kfu * cholesky_solve(Luu, u - phi_z * eta_omega);
}




model {
    eta_omega ~ std_normal();
    eta ~ std_normal();
}

generated quantities {
        vector[N] prior;
        vector[N] update;
        vector[N] posterior;
        vector[M] sample_from_kernel;
        vector[M] sample_from_rff;
        
        prior = phi_x * eta_omega;
        update = Kfu * cholesky_solve(Luu, u - phi_z * eta_omega);
        posterior = prior + update;
        
        
        
        sample_from_kernel = Luu * eta;
        sample_from_rff = phi_z * eta_omega;
}

"""

N = 50
N_pred = 10
M = 10
D = 1 # Stan only supports the case D = 1 (in which input is a vector)
x = np.linspace(-2,2, N)
z = np.linspace(-2,2, M)
x_pred = np.linspace(-2.5, 2.5, N_pred)
y = np.random.randn(N)
u = np.random.randn(M)

num_feature = 100
omega = np.random.randn(D, num_feature)

data = {'N': N,
        'N_pred': N_pred,
        'M': M,
        'D': D,
        'x': x,
        'y': y,
        'z': z,
        'u': u,
        'x_pred': x_pred,
        'num_feature':num_feature,
        'omega':omega,
        'length_scale':0.1,
        'alpha':1.,
        'sigma':0.01
        }

model = create_model(model_code=stan_code)

fit = model.sampling(data=data, iter=10, chains=4)

output = fit.extract(permuted=True)

prior = output['prior']
update = output['update']
posterior = output['posterior']

plt.figure()
for i in range(10):
        plt.plot(x, prior[i,:])
        plt.title("prior")

plt.figure()
for i in range(10):
        plt.plot(x, update[i,:])
        plt.title("update")

plt.figure()
plt.plot(z, u, 'k.')
for i in range(10):
        plt.plot(x, posterior[i,:])
        plt.title("posterior")

plt.figure()
for i in range(10):
        plt.plot(z, output['sample_from_kernel'][i,:])

plt.figure()
for i in range(10):
        plt.plot(z, output['sample_from_rff'][i,:])

plt.show()
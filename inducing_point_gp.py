import pystan
import numpy as np
import matplotlib.pyplot as plt

stan_code = """

data{
    int<lower=1> N;  // number of data
    int<lower=1> N_pred; // number of test
    int<lower=1> M; // number of inducing points
    int<lower=1> D; // number of dimensions
    vector[N] y;
    real x[N];
    real z[M]; //let's fix inducing point location
    real x_pred[N_pred]; 
}

parameters {
    // kernel parameters
    real<lower=0> length_scale;
    real<lower=0> alpha;
    real<lower=0> sigma;
    // inducing points
    // real z[M]; //location
    vector[M] eta;
    
    vector[N] epsilon;
}

transformed parameters {
    real jitter = 1e-6;
    matrix[M, M] Kuu;
    matrix[M, M] Luu;
    matrix[N, M] Kfu;
    matrix[N, N] Kff;
    vector[M] Kuu_div_u;
    matrix[M, N] var_pred;
    vector[M] u;
    vector[N] mean_f;
    matrix[N, N] covar;
    matrix[N, N] Lcovar;
    Kuu = cov_exp_quad(z, alpha, length_scale);
    Kfu = cov_exp_quad(x, z, alpha, length_scale);
    Kff = cov_exp_quad(x, alpha, length_scale);
    for (m in 1:M)
        Kuu[m,m] = Kuu[m,m] + jitter;
    Luu = cholesky_decompose(Kuu);
    u = Luu * eta;
    
    Kuu_div_u = mdivide_left_tri_low(Luu, u);
    Kuu_div_u = mdivide_right_tri_low(Kuu_div_u', Luu)';
    mean_f = Kfu * Kuu_div_u;
    var_pred = mdivide_left_tri_low(Luu, Kfu');
    covar = Kff - var_pred'*var_pred;
    Lcovar = cholesky_decompose(covar + jitter);
}

model {
    
    // hyperparameter
    //length_scale ~ inv_gamma(2,20);
    //alpha ~ inv_gamma(1,5);
    //sigma ~ normal(0,1);
    // for sampling inducing point
    eta ~ normal(0,1); 
    
    epsilon ~ normal(0,1);
    
    //y ~ multi_normal(mean, covar); # this is the problem. Its will be 0(N^3) for this
}

generated quantities {
        // This model is computational unstable (compute the Cholesky of covar). 
        // So just stop at sampling without any further inference
        vector[N] sample;
        sample = mean_f + Lcovar * epsilon;
}

"""

N = 10
N_pred = 10
M = 100
D = 1  # Stan only supports the case D = 1 (in which input is a vector)
x = np.linspace(-2, 2, N)
z = np.linspace(-2,2, M)
x_pred = np.linspace(-2.5, 2.5, N_pred)
y = np.random.randn(N)


data = {'N': N,
        'N_pred': N_pred,
        'M': M,
        'D': D,
        'x': x,
        'y': y,
        'z':z,
        'x_pred': x_pred
        }

model = pystan.StanModel(model_code=stan_code)

fit = model.sampling(data=data, iter=10, chains=4)

output = fit.extract(permuted=True)

sample = output['sample']

for i in range(10):
    plt.plot(x, sample[i, :])

plt.show()
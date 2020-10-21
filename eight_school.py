import pystan

import matplotlib.pyplot as plt

school_code = """
data{
    int<lower=0> J;  // number of schools
    vector[J] y; // estimated treatment effects
    vector<lower=0>[J] sigma;
}

parameters{
    real mu;
    real<lower=0> tau;
    vector[J] eta;
}

transformed parameters{
    vector[J] theta;
    theta = mu + tau * eta;
}

model{
    eta ~ normal(0,1);
    y ~ normal(theta, sigma);
}
"""


sm = pystan.StanModel(model_code=school_code)

schools_dat = {'J': 8,
               'y': [28,  8, -3,  7, -1,  1, 18, 12],
               'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}

fit = sm.sampling(data=schools_dat, iter=1000, chains=4)

la = fit.extract(permuted=True)
mu = la["mu"]
print(mu)

fit.plot()
plt.show()

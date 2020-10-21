## Playing with Stan
This repo contains several implementation of GPs written in Stan

### 1. Efficient GP posterior sampling
A recent paper in ICML  with the reference as
```
@inproceedings{wilson2020efficiently,
    title={Efficiently sampling functions from Gaussian process posteriors},
    author={James T. Wilson
            and Viacheslav Borovitskiy
            and Alexander Terenin
            and Peter Mostowsky
            and Marc Peter Deisenroth},
    booktitle={International Conference on Machine Learning},
    year={2020},
    url={https://arxiv.org/abs/2002.09309}
}
```
This paper has a direct application in sparse GP in the case that we need a sample from posterior with a number of test points. 
Typically, the computation complexity to sample from a multivariate Gaussian is the big-O of the cubic of the number of dimensions.
If the number of test points is big, it becomes impossible to perform sampling.

The paper tackles the this problem by using Matheron's rule to represent the conditional Gaussian 
distribution. According to this rule, the conditional distribution will be divided into two parts: 
prior sampling which will sample from weight space and update which is derived from functional view.

The correspond implementation in Stan can be found in file ``gp_sampling.py``.

There are two important parts to implement the proposed approach

- Random Fourier Feature (RFF) to sample prior: Given a kernel (here it is squared exponential kernel), we obtain the 
corresponding feature map. A sample can be obtained via the dot-product of this feature map and normal-distributed vector. 
Here is the Stan code to extract feature map

```
matrix rff(real[] x, matrix omega, real length_scale, real alpha){
                int D = rows(omega);
                int num_feature = cols(omega);
                int N = size(x);
                matrix[N, num_feature] phi_x;
                real scale = sqrt(0.5/N);
                phi_x = cos(to_matrix(x, N, D) * omega / length_scale);
                return alpha * phi_x * scale;
        }
```
- Update from functional space: The update is not derived from weight space due to variance starvation. Instead, the kernel function 
involves in this step. If you are familiar with GP, the implementation is not hard to grasp. We need a helper function to 
solve linear system (in taking Cholesky as input)

```
vector cholesky_solve(matrix Luu, vector y){
                int M = rows(Luu);
                vector[M] ret;
                ret = mdivide_left_tri_low(Luu, y);
                ret = mdivide_right_tri_low(ret', Luu)';
                return ret;
        }
```
Accoring the update rule, the update term can be implemented in Stan as
```
Kfu * cholesky_solve(Luu, u - phi_z * eta_omega);
```
which is the second component in the sum of Equation (13) in the paper.

There are still some problems in the way of implementing _sparse_ Gaussian process. For example,
the inducing points are learned from variational inference (or optimization). The question is that: can we learning this with sampling
techniques in Stan?




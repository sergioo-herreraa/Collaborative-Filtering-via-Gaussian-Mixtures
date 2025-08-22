"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from scipy.stats import multivariate_normal

def observed_values_Cu(X, u, j, mixture):
    x = X[u, :]                        
    mask = x != 0
    Cu = np.where(mask, 1, 0)

    x_cu = x[mask]
    size = len(x_cu)                   
    miu_cu = mixture.mu[j, mask]     
    var_cu = mixture.var[j]*np.identity(size, size) 

    return Cu, x_cu, miu_cu, var_cu

def f_u_j(X, u, j, mixture):
  Cu, x_cu, miu_cu, var_cu = observed_values_Cu(X, u, j, mixture)
  gaussian = multivariate_normal.pdf(x_cu, miu_cu, var_cu)
  return np.log(mixture.p[j])+np.log(gaussian)

def log_pj_given_u(X, u, j, mixture):
  K = len(mixture.var)
  sum_exp = np.sum([np.exp(f_u_j(X, u, i, mixture)) for i in range(K)])
  return f_u_j(X, u, j, mixture)-np.log(sum_exp)

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K = len(mixture.var)
    p = np.zeros((n, K))
    log_likelihood_n=0

    for u in range(n):
        log_likelihood_k= 0
        for j in range(K):
            p[u, j] = log_pj_given_u(X, u, j, mixture)
            log_likelihood_k += f_u_j(X, u, j, mixture)

        log_likelihood_n += log_likelihood_k

    return np.exp(p), log_likelihood_n
    

def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """ 
    n, d = X.shape
    K = len(mixture.var)
    miu= mixture.mu
    var = mixture.var
    p = mixture.p

    for k in range(K):
      for l in range(d):
        num_mu = 0
        denom_mu= 0
        num_var = 0
        denom_var = 0
        p_index = 0
        for u in range(n):
          Cu, x_cu, miu_cu, var_cu = observed_values_Cu(X, u, k, mixture)
          num_mu += post[u, k]*Cu[l]*X[u,l]
          denom_mu += post[u, k]*Cu[l]
          num_var += post[u, k]*np.abs(np.linalg.norm(x_cu-miu_cu))**2
          denom_var += post[u, k]*len(x_cu)
          p_index+=post[u, k]

        if denom_mu >=1:
            miu[k,l] = num_mu/denom_mu
          
        var[k] = max(num_var/denom_var, min_variance)

        p[k] = p_index/n

    return GaussianMixture(miu, var, p)
    
          
def run(X: np.ndarray, mixture: GaussianMixture) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_log_likelihood= None
    log_likelihood= None

    while (log_likelihood is None or log_likelihood - old_log_likelihood > 1e-6*abs(log_likelihood)):
      old_log_likelihood = log_likelihood
      post, log_likelihood = estep(X, mixture)
      mixture = mstep(X, post, mixture)

    return mixture, post, log_likelihood



def fill_matrix(X: np.ndarray, mixture: GaussianMixture, K, seed) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    mixture, post = common.init(X, K, seed)
    mixture, post, log_likelihood = run(X, mixture)
    
    for u in range(n):
        for l in range(d):
            if X[u, l]==0:
                k=np.argmax(post[u, :])
                X[u, l]=mixture.mu[k, l]
    return X

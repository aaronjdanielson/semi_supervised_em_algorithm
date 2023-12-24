import sys
import numpy as np
from scipy.stats import multivariate_normal


#####################################################################
## let's write a function to apply the multivariate normal to a matrix
#####################################################################
def est_mult_gaus(X,mu,sigma):
    p = multivariate_normal.pdf(X, mean=mu, cov=sigma, allow_singular = True)
    p[p == 0.0] = sys.float_info.min
    return p

#####################################################################
## write a function to upate the zhats
#####################################################################
def get_zhats(mymat,mu1,mu0,cov1,cov0,mypi):
    p1 = est_mult_gaus(mymat,mu1,cov1)
    p0 = est_mult_gaus(mymat,mu0,cov0)
    zhat = mypi*p1/(mypi*p1 + (1-mypi)*p0)
    return zhat

#####################################################################
## write a function to update mu
#####################################################################
def get_mu(mu_group,mat_unsup,zhat,mylambda,Ngroup,Nobs,Nmis):
    num = (Ngroup*mu_group) + (mylambda) * np.sum((mat_unsup.T * zhat).T,axis = 0)
    denom =  Ngroup + (mylambda)*sum(zhat)
    mu = num/denom
    return mu

#####################################################################
## write a function to update sigma
#####################################################################

def get_deviation(mat,mu,zhat):
    tmp = mat - mu
    tmp1 = (tmp.T * zhat).T
    return np.dot(tmp1.T, tmp.conj())

def get_sigma(mat_group,mu_group_cur,mat_unsup,zhat,mylambda,Ngroup,Nobs,Nmis):
    num =  get_deviation(mat_group,mu_group_cur,1) + (mylambda)*get_deviation(mat_unsup,mu_group_cur,zhat)
    denom =  Ngroup + (mylambda)*sum(zhat)
    sigma = num/denom
    return sigma

#####################################################################
## write a function to update pi
#####################################################################

def get_pi(Ngroup,Nobs,Nmis,mylambda,zhat):
    num = Ngroup + mylambda * np.sum(zhat)
    denom = Nobs + mylambda * Nmis
    return  num/denom #(1/Nobs) * Ngroup + (mylambda/Nmis)*np.sum(zhat)



import sys
import numpy as np
from scipy.stats import multivariate_normal
import em_functions as em

#####################################################################
## em algorithm
#####################################################################

def em_algorithm(mylambda,df_true,df_false,df_unsup,mu1,mu1_init,mu0,mu0_init,Sigma1_init,Sigma0_init,pi_init,cutoff):
    N1 = df_true.shape[0]
    N0 = df_false.shape[0]
    Nobs = N1 + N0
    N_unsup = df_unsup.shape[0]

    pi_cur = pi_init
    mu1_cur = mu1_init
    mu0_cur = mu0_init
    Sigma1_cur = Sigma1_init
    Sigma0_cur = Sigma0_init
    i = 0
    eps = 1
    while eps > cutoff:
        i += 1
        print(i)
        #####################################################################
        #### Expectation Step
        #### update the posterior probabilities of the unsupervised
        #####################################################################
        zhats_new = get_zhats(df_unsup,mu1_cur,mu0_cur,Sigma1_cur,Sigma0_cur,mypi = pi_cur)
        print("updated zhat")
        #####################################################################
        ## Maximization Step
        #### update the model parameters
        #####################################################################
        pi_new = get_pi(N1,Nobs,N_unsup,.5,zhats_new)
        print("updated pi")
        mu1_new = get_mu(mu1,df_unsup,zhats_new,mylambda,N1,Nobs,N_unsup)
        mu0_new = get_mu(mu0,df_unsup,(1-zhats_new),mylambda,N0,Nobs,N_unsup)
        print("updated mu")
        Sigma1_new = get_sigma(df_true,mu1_new,df_unsup,zhats_new,mylambda,N1,Nobs,N_unsup)
        Sigma0_new = get_sigma(df_false,mu0_new,df_unsup,(1-zhats_new),mylambda,N0,Nobs,N_unsup)
        print("updated Sigma")
        #####################################################################
        #### see how much things change
        #####################################################################
#####################################################################
        res = max(np.max(abs(mu1_new - mu1_cur)), np.max(abs(mu0_new - mu0_cur)),np.max(abs(Sigma1_new - Sigma1_cur)),np.max(abs(Sigma0_new - Sigma0_cur)))
        zhats_cur = zhats_new
        pi_cur = pi_new
        mu1_cur = mu1_new
        mu0_cur = mu0_new
        Sigma1_cur = Sigma1_new
        Sigma0_cur = Sigma0_new
        print(res)
        eps = res
    return (zhats_cur,mu1_cur,mu0_cur,Sigma1_cur,Sigma0_cur,pi_cur)


#####################################################################
#####################################################################
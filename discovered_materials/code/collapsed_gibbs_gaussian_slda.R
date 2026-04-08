#####################################################
#####################################################
## Gaussian sLDA
## collapsed Gibbs Monte Carlo for slda
#####################################################
#####################################################

#####################################
## set hyperparameters: alpha and eta
#####################################
K <- 5 ## number of topics
Mv <- ## dimension of the word embedding
alpha <- 1.0
mu <- numeric(Mv) ## prior for topic mean
kappa <- 1   ## strength of the prior mean
Sigma <- diag(Mv) ## 
Psi <- diag(K)  ## topic covariance
nu <- 1  ## strength of prior topic covariance


## load the word2vec embeddings

## do the fast cholesky sampling

## add the supervised part


###############################################################################
## step 1:  initialize the algorithm by randomly assigning a topic to each word
##          and initialize the counts needed to sample the topics
###############################################################################

## initialize: n^k_i = 0, n_i = 0, n_k^t = 0, n_k = 0


## number of times any word from document d is assigned to topic k
## n_{d,k}
doc_topic <- matrix(0,D,K)     ## in the numerator with alpha.
## total number of words in document d
## n_d                                
doc_topic_sum <- numeric(D)  ## in the denominator with alpha
## number of times word v has been assigned to topic k
## n_{v,k}
topic_word <- matrix(0,V,K)
## number of words assigned to topic k
## n_k
topic_word_sum <- numeric(K)

## average vectors
vbars <- matrix(0,K,Mv)

## empirical sample covariance matrices
Cs <- array(0,c(Mv,Mv,K))


################################################
## run a pre-loop to initialize the counts
################################################

for (d in c(1:D)){
  for (l in c(1:WW[d])){
    ################################################
    ##### A: Initialize the topic for token d,l ####
    ################################################
    ## sample z_{i,l} ~ Mult(1/K)
    topic[[d]][1,l] <- tmp <- sample(1:K,1)
    ################################################
    ##### A: Initialize the count objects ##########
    ################################################
    ## (1) increment document-topic count: n^k_i = n^k_i + 1  ## need this
    ## n^k_d = the number of times any word from document d has been assigned to topic k
    doc_topic[d,tmp] <- doc_topic[d,tmp] + 1
    ## (4) increment topic-word sum: n_k = n_k + 1
    topic_word_sum[tmp] <- topic_word_sum[tmp] + 1 
    
    ## update vbar 
    vbars[tmp,] <- (vbars[tmp,] + words[[d]][l,])/topic_word_sum[tmp]
    
    ## update Cs
    Cs[,,tmp] <- 
    
    
    ## (2) increment document-topic sum: n_i = n_i + 1
    ## n_i = number times ?
    #doc_topic_sum[d] <- doc_topic_sum[d] + 1
    ## (3) increment topic-word count:  n^t_k = n^t_k + 1
    ## n^v_k = number of times word v has been assigned to topic k in any document
    #v <- words[[d]][l]
    #topic_word[v,tmp] <- topic_word[v,tmp] + 1  
    ## (4) increment topic-word sum: n_k = n_k + 1
    topic_word_sum[tmp] <- topic_word_sum[tmp] + 1 
  }
}

#################################
## create the objects vbar_k, C_k
#################################

## average vector scores
vbars <- 
Cs <- 
  
#kappas <- numeric(K)
kappas <-  kappa + topic_word_sum
nus <-  nu + topic_word_sum

#nu <- numeric(K)
mus <- (kappa * mu + topic_word_sum * vbars)/kappas

  #matrix(0,K,Mv)
Sigmas <- array(0,c(Mv,Mv,K))
Psis <-  array(0,c(Mv,Mv,K))


##############################################################################
##############################################################################
##########################################################
## E-Step: Draw new topics for each word v_{d,l} R times
##########################################################
for (r in c(2:R)){
  for (d in c(1:D)){  # for all documents
    for (l in c(1:WW[d])){  # for all tokens l in document d
      ####################################################################################
      ## get current topic assignment for token l in document d index
      ###################################################################################
      tmp <- topic[[d]][(r-1),l]
      ############################
      ##### A: decrement step ####
      ############################
      ## there are four numbers to update as in the initialization step
      ## for the current assignment of k to term _ for word w_{i,l}
      ## (1)
      doc_topic[d,tmp] <- doc_topic[d,tmp] - 1
      ## (2)
      doc_topic_sum[d] <- doc_topic_sum[d] - 1
      ## (3)
      v <- words[[d]][l]
      topic_word[v,tmp] <- topic_word[v,tmp] - 1
      ## (4)
      topic_word_sum[tmp] <- topic_word_sum[tmp] - 1 
      ###########################
      ##### B: sampling step #####
      ############################
      #########################################################
      ## let's compute the empirical documentation proportions
      tmp_topic <- topic[[d]][(r-1),]
      counts <- sapply(1:K, function(x){
        sum(tmp_topic == x)
      })
      counts[tmp] <- counts[tmp] - 1
      new_mat <- repmat(counts,n=K,m  = 1) + diag(K)
      zhat <- new_mat / length(words[[d]])
      ## remove the current 
      #tmp_topic <- tmp_topic[-l]
      #zhat <- xtabs(~topic[[d]][(r-1),])
      #########################################################
      ## compute weight vector
      ## wtvec = (c^{-}_{v,k} + eta)/(c^{-}_k + V*eta) * (c^{-}_{i,k} + alpha)/(L_i + K*alpha)
      ## the weight vector has length K
      pw <- (topic_word[v,] + eta)/(topic_word_sum + eta*V)
      pz <- (doc_topic[d,] + alpha)/(doc_topic_sum[d] + alpha*K)
      py <- dnorm(rep(response[d],K),zhat%*%cur_eta+ rep(structured_fields[d,]%*%cur_delta,K), sd = cur_sd)
      wtvec <- pw*pz*py
      ########################################################
      ## sample a topic assignment
      ########################################################
      topic[[d]][r,l] <- tmp <- sample(c(1:K),1,prob = wtvec)
      
      ############################
      ##### C: increment step ####
      ############################
      ## (1) increment document-topic count: n^k_i = n^k_i + 1
      ## n^k_i = the number of times any word from document i has been assigned to topic k
      doc_topic[d,tmp] <- doc_topic[d,tmp] + 1
      ## (2) increment document-topic sum: n_i = n_i + 1
      ## n_i = number times ?
      doc_topic_sum[d] <- doc_topic_sum[d] + 1
      v <- words[[d]][l]
      ## (3) increment topic-word count:  n^t_k = n^t_k + 1
      ## n^v_k = number of times word v has been assigned to topic k in any document
      topic_word[v,tmp] <- topic_word[v,tmp] + 1  
      ## (4) increment topic-word sum: n_k = n_k + 1
      topic_word_sum[tmp] <-topic_word_sum[tmp] + 1 
    }
  }
  print(r)
}

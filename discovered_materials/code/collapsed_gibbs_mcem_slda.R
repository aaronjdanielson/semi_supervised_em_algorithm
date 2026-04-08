#####################################################
#####################################################
## collapsed gibbs Monte Carlo EM for slda
#####################################################
#####################################################

## arguments:  vocab, words, alpha, eta, K, R, response, X
require(pracma)

#####################################
## set hyperparameters: alpha and eta
#####################################
alpha <- 1.0
eta <- 0.1


## helper functions

get_topic_probs <- function(d,rrs){
  tmp <- xtabs(~topic[[d]][rrs,])
  tmp/(sum(tmp))
}

#####################################
#####################################
## load the data
#####################################
#####################################

#####################################
## character vector of unique words
#####################################
vocab <- readRDS("~/Dropbox/ICBC_readings/StatisticalNLP/data/vocab.rds")

###########################################################################
## list of length equal to the number of documents
## each element of the list contains a vector with the index for each word
###########################################################################
words <- readRDS("~/Dropbox/ICBC_readings/StatisticalNLP/data/words.rds")
doc_list <- readRDS("~/Dropbox/ICBC_readings/StatisticalNLP/data/doc_list.rds")

set.seed(444)
structured_fields <- readRDS("~/Dropbox/ICBC_readings/StatisticalNLP/data/structured_fields.rds")
structured_fields <- structured_fields + matrix(rnorm(nrow(structured_fields)*ncol(structured_fields)),nrow(structured_fields),ncol(structured_fields))
response <- readRDS("~/Dropbox/ICBC_readings/StatisticalNLP/data/response.rds")
response[which(is.na(response))] <- 10
response <- response + rnorm(length(response))

## set some constants for the algorithm
D <- length(words)  ## number of documents
V <- length(vocab) ## number of distinct words in the vocabulary
WW <- unlist(lapply(words,length)) ## number of tokens in each document (allows for multiple occurences of the same word)
L <- bb ## vector of word counts in each of the documents, e
nX <- ncol(structured_fields)  ## dimension of the structured field

K <- 3 ## number of topics

R <- 2000 ##  number of Gibbs samples

burnin <- 500
thin <- 25
my_threshold <- .05

myrs  <- seq(from = burnin + 1, to = R, by = thin)
M <- length(myrs)

mylist <- list()

## list of length equal to the number of documents
## each element of the list contains a matrix rows = R and cols equal to the number of words
## this contains the assignment of each word to a topic
topic  <- list(D)
for (d in c(1:D)){
  topic[[d]] <- matrix(0,R,WW[d])
}

#getCounts <-  function(vec,myK=K){
#  sapply(1:myK, function(x){
#    sum(vec == x)
#  })
#}

## step 1:  initialize the algorithm by randomly assigning a topic to each word
##          and initialize the counts needed to sample the topics

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
    ## (1) increment document-topic count: n^k_i = n^k_i + 1
    ## n^k_i = the number of times any word from document i has been assigned to topic k
    doc_topic[d,tmp] <- doc_topic[d,tmp] + 1
    ## (2) increment document-topic sum: n_i = n_i + 1
    ## n_i = number times ?
    doc_topic_sum[d] <- doc_topic_sum[d] + 1
    ## (3) increment topic-word count:  n^t_k = n^t_k + 1
    ## n^v_k = number of times word v has been assigned to topic k in any document
    v <- words[[d]][l]
    topic_word[v,tmp] <- topic_word[v,tmp] + 1  
    ## (4) increment topic-word sum: n_k = n_k + 1
    topic_word_sum[tmp] <-topic_word_sum[tmp] + 1 
  }
}


######################################
## initialize the parameter estimates
######################################

etas_init <- rnorm(K)
deltas_init <- coef(lm(response ~ structured_fields + 0 ))
sd_init <- 1

cur_eta <- etas_init
cur_delta <- deltas_init
cur_sd <- sd_init

eps <- 1

iter <- 1

mylist[[iter]] <- list(eta = cur_eta,delta = cur_delta, sd = cur_sd)

while(eps > my_threshold){

  #cur_eta <- etas_init
  #cur_delta <- deltas_init
  #cur_sd <- sd_init




#while(eps > my_threshold){
  ##########################################################
  ## E-Step: Draw new topics for each word w_{d,l} R times
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
  ##########################################################
  ## M-Step: Maximize the Approximated Log-Likelihood
  ##########################################################
  ## get estimates for eta and delt
  
  
  bigZ <- NULL 
  for (rr in myrs){
    design_mat <- t(sapply(c(1:D), function(x){
      sapply(1:K,function(y){
        sum(topic[[x]][rr,] == y)
      })
                         }
      ))
    bigZ <- rbind(bigZ,design_mat)
  }
  bigZ <- bigZ/rowSums(bigZ)
    
  bigY  <- rep(response,M)
  bigX  <- repmat(structured_fields,n  = M, m = 1)
  
  res <- lm(bigY ~ bigZ + bigX + 0)
  new_eta <- coef(res)[1:K]
  new_delta <- coef(res)[-(1:K)]
  new_sd <- summary(res)$sigma
  iter <- iter + 1
  mylist[[iter]] <- list(eta = new_eta,delta = new_delta, sd = new_sd)
  
  eps <- max(c(abs(new_eta- cur_eta),abs(new_delta - cur_delta),abs(new_sd - cur_sd)))
  print(eps)
  cur_eta <- new_eta
  cur_delta <- new_delta
  cur_sd <- new_sd
  
  
  ## make the big design matrices
  ## the first K features are the topics
  ## the next group are the structured fields
  ##################################################
  ##################################################
  ## next:  update the counts
  ##################################################
  ##################################################
  ################################################
  ## run a pre-loop to initialize the counts
  ################################################
  
  for (d in c(1:D)){
    for (l in c(1:WW[d])){
      ################################################
      ##### A: Initialize the topic for token d,l ####
      ################################################
      ## sample z_{i,l} ~ Mult(1/K)
      topic[[d]][1,l] <- topic[[d]][R,l]
    }
  }
  
  
}


saveRDS(mylist,"~/Dropbox/ICBC_readings/StatisticalNLP/data/mcem_results.rds")
#saveRDS(mylist,"~/Dropbox/ICBC_readings/StatisticalNLP/data/mcem_results.rds")


get_top_topics  <- function(k){
  vocab[order(topic_word[,k],decreasing = T)[1:10]]
}

sapply(c(1:5),get_top_topics)


plot(c(1:2000),topic[[1]][c(1:2000),4],col = "blue",pch = 20, cex = .25)
plot(c(1:5000),topic[[1]][c(1:5000),3],col = "blue",pch = 20, cex = .25)
points(c(1:5000),topic[[1]][c(1:5000),3],col = "purple",pch = 20, cex = .25)

res_init <- lm(response ~ structured_fields + 0 )






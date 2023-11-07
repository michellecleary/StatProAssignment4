#inital framework

netup <- function(d){
  # Function to represent the network given by nodes divded by layers.
  # Inputs:
  # d: vector of nodes in each level of the network
  # Outputs:
  # h: list of nodes in each layer of the network
  # W: list of corresponding weight matrices to link each layer to the one above it
  # b: List of offset vectors linking each layer to the one above it
  h <- list()
  for (i in 1:length(d)) {
    h[[i]] <- rep(1, d[i])
  }
  
  W <- list()
  b <- list()
  for (i in 1:(length(d) - 1)) {
    W[[i]] <- matrix(runif(d[i + 1] * d[i], max = 0.2), nrow=d[i + 1], ncol = d[i])
    b[[i]] <- runif(d[i + 1], max = 0.2)
  }

  result <- list("h"=h,
                "W"=W,
                "b"=b)  
  return(result)
}

forward <- function(nn,inp){
  #Function description
  #Inputs:
  #nn: list containing infromation on the nodes and layers of the network as described in netup
  #imp: input values for the first layer of the network
  #Outputs:
  #nn: updated list of node values on each layer
  h<-nn$h
  W<-nn$W
  b<-nn$b
  
  h[[1]]<-inp
  
  for (i in 1:(length(h)-1)){
    for (j in 1:length(h[[i+1]])){
  h[[i+1]][j]<-max(0,W[[i]][j, ]%*%h[[i]]+b[[i]][j])
    }}
  result<-list("nn"=h)
}

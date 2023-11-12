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
  #Computes the rests of the node values for an inputted initial layer 1 values.
  #Inputs:
  #nn: list containing infromation on the nodes and layers of the network as described in netup
  #imp: input values for the first layer of the network
  #Outputs:
  #nn: updated list of matrices of node values on each layer
  h<-nn$h
  W<-nn$W
  b<-nn$b
  
  h[[1]]<-inp
  
  for (i in 1:(length(h)-1)){
    for (j in 1:length(h[[i+1]])){
  h[[i+1]][j]<-max(0,W[[i]][j, ]%*%h[[i]]+b[[i]][j])
    }}
  result<-list("h"=h,
               "W"=W,
               "b"=b)
  return(result)
}


backward <- function(nn, k){
  # Function to compute the derivatives of the loss corresponding to output 
  # class k for network nn.
  #
  # Inputs:
  # nn: network list as output from forward()
  # k: output class
  #
  # Outputs:
  # updated_net: updated network list with the following elements:
  #              h: a list of nodes for each layer
  #              W: a list of weight matrices
  #              b: a list of offset vectors
  #              dh: derivatives with respect to the nodes
  #              dW: derivatives with respect to the weights
  #              db: derivatives with respect to the offsets
  
  # Extract h, W and b from input
  h <- nn$h
  W <- nn$W
  b <- nn$b
  
  # Number of layers in network
  L <- length(h)
  
  # Compute number of nodes per layer
  nodes_per_layer <- rep(0, L)
  for (layer in 1:L){
    nodes_per_layer[layer] <- length(h[[layer]])
  }
  
  # Initialise list to store derivatives w.r.t nodes for each layer
  dh <- list()
  for (i in 1:L) {
    dh[[i]] <- rep(0, nodes_per_layer[i])
  }
  
  # Compute the derivative of the loss for k w.r.t nodes in final layer, L
  
  # Sum of exponential of each node value in final layer
  #sum_exp_final_layer <- sum(exp(h[[L]]))
  # 
  # # Iterate over nodes in final layer
  # for (j in 1:nodes_per_layer[L]){
  #   # Check whether the node corresponds to class k and assign appropriate
  #   # derivative value for that node
  #   if (j == k){
  #     dh[[L]][j] <- (exp(h[[L]][j]) / sum_exp_final_layer) - 1
  #   }
  #   else if (j != k){
  #     dh[[L]][j] <- exp(h[[L]][j]) / sum_exp_final_layer
  #   }
  # }
  
  
  # Derivative for nodes in final layer
  dh[[L]] <- exp(h[[L]]) / sum(exp(h[[L]]))
  # Assign appropriate value to node for kth class
  dh[[L]][k] <- dh[[L]][k] - 1
  
  # Compute the derivatives of the loss with respect to the nodes in all other
  # layers using back-propagation
  
  # Initialise vector d
  d <- list()
  for (i in 1:L) {
    d[[i]] <- rep(0, nodes_per_layer[i])
  }
  d[[L]] <- ifelse(h[[L]] > 0, dh[[L]], 0)
  
  # Iterate backwards over each layer, starting at 2nd last layer
  for (l in (L - 1):1){
    
    # Iterate over nodes in next layer
    for (j in 1:nodes_per_layer[l + 1]){
      # Check whether the node has a positive value
      if (h[[l + 1]][j] > 0){
        # Assign the corresponding element of d with the derivative of the loss
        # w.r.t this node
        d[[l + 1]][j] <- dh[[l + 1]][j]
      }
      # If node does not have a positive value
      else if (h[[l + 1]][j] <= 0){
        # Assign the corresponding element of d with a value of 0
        d[[l + 1]][j] <- 0
      }
    }
  }
  
  
  # Iterate backwards over each layer, starting at 2nd last layer
  for (l in (L - 1):1){
    dh[[l]] <-t(W[[l]]) %*% d[[l + 1]]
    d[[l]] <- ifelse(h[[l]] > 0, dh[[l]], 0)
  }
  
  # Initialise lists to store derivatives w.r.t offset vectors and weights 
  # for layers 1 to L - 1
  db <- list()
  dW <- list()
  for (i in 1:(L - 1)){
    db[[i]] <- rep(0, nodes_per_layer[i + 1])
    dW[[i]] <- matrix(0, nrow = nrow(W[[i]]), ncol = ncol(W[[i]]))
  }
  
  # Iterate over layers L-1 to 1
  for (l in (L - 1):1){
    # Assign the offset vectors derivatives with the values of d^(l + 1) 
    db[[l]] <- d[[l + 1]]
    # Assign the weights derivatives with the values of d^(l + 1) (h^l)^T
    dW[[l]] <- d[[l + 1]] %*% t(h[[l]])
  }
  
  # Update network list
  updated_network_list <- list('h' = h,
                               'W' = W,
                               'b' = b,
                               'dh' = dh,
                               'dW' = dW,
                               'db' = db)
  
  return (updated_network_list)
}


train<-function(){
  #Loop over the nsteps desired
  #First take a new sample of the data( mb points)
  #Compute the network for each of the points with W and b
  #Then compute the gradients for all of these points
  #Average over all of them
  #Take a step to update W and b
  #end of loop
  #Then once the optimum W and b have been found find networks for all of the data points
  #The final nodes of these will be fed into a classification function which will tell us which class the thing should enter
}




train <- function(nn,inp,k,eta=.01,mb=10,nstep=10000){
  # Extract h, W, and b from input
  h <- nn$h
  W <- nn$W
  b <- nn$b
  
  # Number of layers in network
  L <- length(h)
  
  # Compute number of nodes per layer
  nodes_per_layer <- rep(0, L)
  for (layer in 1:L){
    nodes_per_layer[layer] <- length(h[[layer]])
  }
  
  # Ordered class labels 
  class_labels <- sort(unique(k))
  number_classes <- length(class_labels)
  
  # Iterate over each step
  for (step in 1:nstep){
    
    # Randomly sample mb rows of data
    index <- sample(1:nrow(inp), mb, replace=FALSE)
    sampled_inp <- inp[index, ]
    sample_class <- k[index]
    
    # Initialise sum of gradients for all mb data points
    all_db <- list()
    all_dW <- list()
    
    for (i in 1:(L - 1)){
      all_db[[i]] <- rep(0, nodes_per_layer[i + 1])
      all_dW[[i]] <- matrix(0, nrow = nrow(W[[i]]), ncol = ncol(W[[i]]))
    }
    
    # Iterate over all mb sampled rows
    for (samp in 1:nrow(sampled_inp)){
      
      # Update network using forward and then back-propagation
      nn <- forward(nn, sampled_inp[samp, ])
      back_prop_gradients <- backward(nn, k = sample_class[samp])
      
      # Iterate over layers 1 to L - 1
      for (l in 1:(L - 1)){
        # Sum the current gradient values to the overall gradients
        all_dW[[l]] <- all_dW[[l]] + back_prop_gradients$dW[[l]]
        all_db[[l]] <- all_db[[l]] + back_prop_gradients$db[[l]]
      }
    }
    
    # Iterate over layers 1 to L - 1
    for (i in 1:(L - 1)){
      # Update W and b using the average of the computed gradients
      nn$b[[i]] <- nn$b[[i]] - eta *(all_db[[i]] / mb)
      nn$W[[i]] <- nn$W[[i]] - eta * (all_dW[[i]] / mb)
    }
  }
  
  # Return the trained network
  return (nn)
}

# Load the data
data(iris)
# Create a mask which is sequence starting at 5 and ending at the last row index
# of the data with an increment of 5
mask <- seq(5, nrow(iris), 5)
# Use this mask to select every 5th row of the data, starting from row 5, as the
# test data
test_data <- iris[mask, ]
# The training data is all of the data with the test data removed
training_data <- iris[-mask, ]

# Define the initial input values for training as all columns of the training 
# data except the class labels
# Convert to a matrix and unname the columns for use in functions
training_inp <- unname(as.matrix(training_data[, 1:4]))
# Define the classes of the input
# Convert to a vector and unname the column for use in functions
training_classes <- unname(as.vector(training_data$Species))
# Convert string class names to numeric values for use in functions
training_classes_numeric <- as.numeric(factor(training_classes))

# Set the seed
set.seed(4)
# Initialise the network
nn <- netup(c(4, 8, 7, 3))
# Train the network
nn <- train(nn, inp = training_inp, k = training_classes_numeric)




# Michelle Cleary (s1979093), Murad Magdiyev (s2533467), Liz Howell (s1523887)
# https://github.com/michellecleary/StatProAssignment4.git
# Contributions:
# Roughly equal contributions - about 1/3 each.
# Michelle - backward(), train(), prediction for iris data.
# Murad - netup() function, debugging and optimising forward().
# Liz - forward() function, debugging and optimising backward() and train().


# The following code will first define functions to set up a simple neural 
# network for classification, and to train it using stochastic gradient descent. 
# It will then use a training subset of the iris dataset to train a network to 
# classify irises to species based on the 4 characteristics given. The 
# performance of the network will then be evaluated using the misclassification 
# rate (i.e. the proportion misclassified) for the test subset pf the iris 
# dataset.


#### Function definitions


# A simple, fully-connected L-layer neural network of 
netup <- function(d){
  # Function to create a list representing the neural network with d[l] nodes 
  # in layer l.
  #
  # Inputs:
  # d: vector giving the number of nodes in each layer of the network
  #
  # Outputs:
  # nn: network list containing the following elements:
  #     h: list of node values in each layer of the network
  #     W: list of corresponding weight matrices to link each layer to the next 
  #        one 
  #     b: list of offset vectors linking each layer to the next one
  #     L: total number of layers in network
  #     nodes_per_layer: vector where the lth element is the number of nodes in
  #                      layer l of the network
  
  # Initialise empty lists to store h, W, and b
  h <- list()
  W <- list()
  b <- list()
  
  # Define the total number of layers, L
  L <- length(d)
  
  # Iterate over each layer
  for (l in 1:L) {
    # Create the nodes in layer l, initialising the values with NA
    h[[l]] <- rep(NA, d[l])
  }
  
  # Iterate over layers 1 to L - 1
  for (l in 1:(L - 1)) {
    # Create the weight matrix W[[l]] and offset vector b[[l]], both linking 
    # layer l to layer l + 1, initialising the elements with U(0, 0.2) random 
    # deviates
    W[[l]] <- matrix(runif(d[l + 1] * d[l], max = 0.2), nrow=d[l + 1], 
                     ncol = d[l])
    b[[l]] <- runif(d[l + 1], max = 0.2)
  }
  
  nn <- list("h" = h,
             "W" = W,
             "b" = b,
             "L" = L,
             "nodes_per_layer" = d)
  
  return (nn)
}


# Each node value is determined by linearly combining the values from the nodes 
# in the previous layer and non-linearly transforming the result. The following
# function will compute node values using the ReLU transform, 
# ReLU(z) = max(0, z). Therefore, we have that node j in layer l + 1 is given 
# by:
# h_j^(l + 1) = max(0, W_j^l h^l + b^l),
# where W_j^l is the jth row of weight parameter matrix W^l which links layer l 
# to layer l + 1, h^l is the vector of node values for layer l, and b^l is the
# vector of offset parameters which links layer l to layer l + 1.


forward <- function(nn, inp){
  # Function to update the network list nn by using the ReLU transform to 
  # compute the remaining node values implied by inp, the given input data to 
  # be used as the values for the first layer nodes.
  #
  # Inputs:
  # nn: network list as returned by netup() containing the following elements:
  #     h: list of node values in each layer of the network
  #     W: list of corresponding weight matrices to link each layer to the next 
  #        one 
  #     b: list of offset vectors linking each layer to the next one
  #     L: total number of layers in network
  #     nodes_per_layer: vector where the lth element is the number of nodes in
  #                      layer l of the network
  # inp: vector of input values for the first layer of the network
  #
  # Outputs:
  # updated_nn: updated network list, containing the following elements: 
  #             h: list of updated node values in each layer of the network
  #             W: list of corresponding weight matrices to link each layer to 
  #                the next one (as in input nn)
  #             b: list of offset vectors linking each layer to the next one (as
  #                in input nn)
  #             L: total number of layers in network (as in input nn)
  #             nodes_per_layer: vector where the lth element is the number of
  #                              nodes in layer l of the network (as in input 
  #                              nn)
  
  # Extract h, W, b, and L from input
  h <- nn$h
  W <- nn$W
  b <- nn$b
  L <- nn$L
  
  # Set the values of the first layer nodes to be the values of the input data
  h[[1]] <- inp
  
  # Define the total number of layers, L
  #L <- length(h)
  
  # Iterate over layers 1 to L - 1
  for (l in 1:(L - 1)) {
    # Compute the node values of the next layer using the ReLU transform
    h[[l + 1]] <- pmax(0, W[[l]] %*% h[[l]] + b[[l]])
  }
  
  # Update the network list
  updated_nn <- list("h" = h,
                     "W" = W,
                     "b" = b,
                     "L" = L,
                     "nodes_per_layer" = nn$nodes_per_layer)
  
  return (updated_nn)
}


# For a classification task in which numeric variables are used to predict which 
# class an observation belongs to, the input layer would have a node for each 
# numeric variable, and the output layer, layer L, would have a node for each
# possible class. Then, we define the probability that the output variable is 
# in class k to be:
# p_k = exp(h_k^L) / sum_{j} exp(h_j^L).
# Then, using the negative log-likelihood of a multinomial distribution as the 
# loss function, if we have n training data pairs with input, x_i, and output
# class, k_i, the loss is:
# L = -sum_{i = 1,..,n} log(p_{k_i}) / n.
# Stochastic gradient descent involves minimising L by repeatedly finding 
# the gradient of the loss function, L, with respect to the parameters, and
# adjusting the parameters by taking a step in the direction of the negative 
# gradient. The following function will compute the gradient of the loss 
# function with respect to the nodes (h), weights (W), and offsets (b) for a 
# given output class. 
#
# The derivative of the loss for k_i with respect to the nodes in the final
# layer is given by:
# dL_i/dh_j^L = exp(h_j^L) / sum_{q} exp(h_q^L) - 1 when j = k_i, and
# dL_i/dh_j^L = exp(h_j^L) / sum_{q} exp(h_q^L) otherwise.
# The derivatives of L_i with respect to all the other h_j^l can be computed 
# using back-propagation as:
# dL_i/dh^l = (W^l)^T d^(l + 1), where d_j^(l + 1) = dL_i/dh_j^(l + 1) if
# h_j^(l + 1) > 0, and d_j^(l + 1) = 0 if h_j^(l + 1) <= 0.
# The derivatives of L_i with respect to the weights and the offsets are then
# given by:
# dL_i/dW^l = d^(l + 1) (h^l)^T; dL_i/db^l = d^(l + 1).


backward <- function(nn, k){
  # Function to compute the derivatives of the loss with respect to the nodes,
  # weights and offsets corresponding to output class k for network nn.
  #
  # Inputs:
  # nn: network list as output from forward() containing the following elements:
  #     h: list of node values in each layer of the network
  #     W: list of corresponding weight matrices to link each layer to the next 
  #        one 
  #     b: list of offset vectors linking each layer to the next one
  #     L: total number of layers in network
  #     nodes_per_layer: vector where the lth element is the number of nodes in
  #                      layer l of the network
  # k: output class
  #
  # Outputs:
  # updated_nn: updated network list containing the following elements:
  #             h: list of node values for each layer
  #             W: list of weight matrices
  #             b: list of offset vectors
  #             L: total number of layers in network
  #             nodes_per_layer: vector where the lth element is the number of 
  #                              nodes in layer l of the network
  #             dh: derivatives with respect to the nodes
  #             dW: derivatives with respect to the weights
  #             db: derivatives with respect to the offsets
  
  # Extract h, W, b, L, and the number of nodes per layer from input
  h <- nn$h
  W <- nn$W
  b <- nn$b
  L <- nn$L
  nodes_per_layer <- nn$nodes_per_layer
  
  # Initialise vector d and the list dh to store derivatives with respect to 
  # the nodes
  dh <- d <- list()
  
  # Compute the derivative of the loss for k with respect to the nodes in final 
  # layer, L
  dh[[L]] <- exp(h[[L]]) / sum(exp(h[[L]]))
  # Assign appropriate value to node for class k
  dh[[L]][k] <- dh[[L]][k] - 1
  
  # Compute values of d for the final layer
  d[[L]] <- ifelse(h[[L]] > 0, dh[[L]], 0)
  
  # Compute the derivatives of the loss with respect to the nodes in all other
  # layers using back-propagation
  
  # Iterate backwards over each layer, starting at 2nd last layer, L - 1
  for (l in (L - 1):1){
    # Compute derivative of loss with respect to the current layer as 
    # (W^l)^T d^(l + 1)
    dh[[l]] <- t(W[[l]]) %*% d[[l + 1]]
    # Compute d^l
    d[[l]] <- ifelse(h[[l]] > 0, dh[[l]], 0)
  }
  
  # Initialise lists to store derivatives with respect to offset vectors and 
  # weights 
  db <- list()
  dW <- list()
  
  # Iterate over layers L - 1 to 1
  for (l in (L - 1):1){
    # Assign the offset vectors derivatives with the values of d^(l + 1) 
    db[[l]] <- d[[l + 1]]
    # Assign the weights derivatives with the values of d^(l + 1) (h^l)^T
    dW[[l]] <- d[[l + 1]] %*% t(h[[l]])
  }
  
  # Update network list
  updated_nn <- list('h' = h,
                     'W' = W,
                     'b' = b,
                     "L" = L,
                     "nodes_per_layer" = nn$nodes_per_layer,
                     'dh' = dh,
                     'dW' = dW,
                     'db' = db)
  
  return (updated_nn)
}


# The idea of stochastic gradient descent is to minimise the loss by 
# repeatedly finding the gradient of the loss function with respect to the 
# parameters for small randomly chosen subsets of the training data, and to 
# adjust the parameters by taking a step in the direction of the negative 
# gradient. The gradient of L is just the average of the gradients of
# L_i = −log(p_{k_i}) for each element of the training data, so we can compute
# the gradient one data point at a time.
# For a single datum i in this small subset, consisting of input data 
# x_i with class k_i, this is done as follows:
# 1. Set h^1 = x_i and compute the remaining node values h_j^l corresponding to 
# this using the ReLU transform, as in forward().
# 2. Compute the derivative of the loss for k_i with respect to the nodes in the 
# final layer, h_j^L, as in backward().
# 3. Compute the derivatives of the loss with respect to all the other h_j^l
# using back-propagation, and the derivatives with respect to the weights and 
# the offsets, as in backward().
#
# Repeat these steps for each datum i in the subset of the training data, and
# compute the derivative of the loss based on this subset by averaging the 
# derivatives for each i in the set. The parameter updates are then of the form:
# W^l <- W^l - eta (dL/dW^l)
# b^l <- b^l - eta (dL/db^l)
# where eta is a step length.
#
# The following function repeats this entire process many times to train a 
# network.


train <- function(nn, inp, k, eta = .01, mb = 10, nstep = 10000){
  # Function to train the network, nn, given input data inp and corresponding 
  # class labels in k; using a step size of eta, randomly sampling mb data to 
  # compute the gradient, and taking nstep optimisation steps.
  #
  # Inputs:
  # nn: network list containing the following elements:
  #     h: list of node values in each layer of the network
  #     W: list of corresponding weight matrices to link each layer to the next 
  #        one 
  #     b: list of offset vectors linking each layer to the next one
  #     L: total number of layers in network
  #     nodes_per_layer: vector where the lth element is the number of nodes in
  #                      layer l of the network
  # inp: matrix of input values for the first layer of the network
  # k: vector of output class labels for each row of the input data (inp)
  # eta: step size to use when updating the parameters W and b (default: 0.01)
  # mb: number of data (rows from inp) to randomly sample to compute the 
  #     gradients (default: 10)
  # nstep: number of optimisation steps to take (default: 10000)
  #
  # Outputs:
  # nn: trained network list containing the following elements:
  #     h: list of node values for each layer
  #     W: list of weight matrices
  #     b: list of offset vectors
  #     L: total number of layers in network (as in input nn)
  #     nodes_per_layer: vector where the lth element is the number of nodes in 
  #                      layer l of the network (as in input nn)
  
  # Extract h, W, b, L, and the number of nodes per layer from input
  h <- nn$h
  W <- nn$W
  b <- nn$b
  L <- nn$L
  nodes_per_layer <- nn$nodes_per_layer
  
  # Iterate over each step
  for (step in 1:nstep){
    
    # Randomly sample mb rows of data
    index <- sample(1:nrow(inp), mb, replace=FALSE)
    sampled_inp <- inp[index, ]
    sample_class <- k[index]
    
    # Initialise lists to store sum of gradients for all mb data points
    all_db <- list()
    all_dW <- list()
    
    # Initialise sum of gradients as 0 for all matrices and offset vectors
    for (l in 1:(L - 1)){
      all_db[[l]] <- rep(0, nodes_per_layer[l + 1])
      all_dW[[l]] <- matrix(0, nrow = nrow(W[[l]]), ncol = ncol(W[[l]]))
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
    for (l in 1:(L - 1)){
      # Update W and b using the average of the computed gradients
      nn$b[[l]] <- nn$b[[l]] - eta *(all_db[[l]] / mb)
      nn$W[[l]] <- nn$W[[l]] - eta * (all_dW[[l]] / mb)
    }
  }
  # Return the trained network
  return (nn)
}


# The value of loss function, L, for a network nn is computed as
# L = -sum_{i=1,..,n} log(p_{k_i}) / n,
# where p_{k_i} is the probability that the output variable for x_i is in 
# class k_i, with
# p_k = exp(h_k^L) / sum_{j} exp(h_j^L).

loss <- function(nn, inp, k){
  # Function to compute value of loss function, L, for a network nn.
  #
  # Inputs:
  # nn: network list containing the following elements:
  #     h: list of node values in each layer of the network
  #     W: list of corresponding weight matrices to link each layer to the next 
  #        one 
  #     b: list of offset vectors linking each layer to the next one
  #     L: total number of layers in network
  #     nodes_per_layer: vector where the lth element is the number of nodes in
  #                      layer l of the network
  # inp: matrix of input values for the first layer of the network
  # k: vector of output class labels for each row of the input data (inp)
  #
  # Outputs:
  # loss: value of the loss function for the given network and data
  
  # Initialise vectors to store probabilities of each input datum being in its
  # true class
  pk <- c()
  
  # Iterate over each row of input data
  for (i in 1:nrow(inp)){
    # Compute node values for data x_i with class k_i
    updated_nn <- forward(nn, inp[i,])
    h <- updated_nn$h
    L <- updated_nn$L
    # Compute the probability that the output variable is in class k_i
    pki <- exp(h[[L]][k[i]]) / sum(exp(h[[L]]))
    pk <- c(pk, pki)
  }
  
  # Compute the loss 
  loss <- round(- sum(log(pk)) / length(pk), 4)
  return (loss)
}




#### Training and testing the performance of a network 

# The functions defined above are now used to train a 4-8-7-3 network to 
# classify irises to species based on the 4 characteristics given in the iris 
# dataset

# Load the data
data(iris)

## Define the training and test data

# Create a mask which is sequence starting at 5 and ending at the last row index
# of the data with an increment of 5
mask <- seq(5, nrow(iris), 5)
# Use this mask to select every 5th row of the data, starting from row 5, as the
# test data
test_data <- iris[mask, ]
# The training data is all of the data with the test data removed
training_data <- iris[-mask, ]

# Define the initial input values for training and testing as all columns of the 
# training data except the class labels and convert to a matrix
training_inp <- as.matrix(training_data[, 1:4])
test_inp <- as.matrix(test_data[, 1:4])
# Define the classes of the input for both training and testing and convert to 
# a vector 
training_classes <- as.vector(training_data$Species)
test_classes <- as.vector(test_data$Species)
# Convert string class names to numeric values for use in functions
training_classes_numeric <- factor(training_classes, 
                                   levels = c('setosa', 'versicolor', 
                                              'virginica'), 
                                   labels = c(1, 2, 3))
test_classes_numeric <- factor(test_classes, 
                               levels = c('setosa', 'versicolor', 'virginica'), 
                               labels = c(1, 2, 3))

## Training the network

# Set the seed
set.seed(13)
# Initialise the network
nn <- netup(c(4, 8, 7, 3))

# Compute pre-training loss using training data
pre_training_loss <- loss(nn, training_inp, training_classes_numeric)
cat('Pre-training loss:', pre_training_loss)

# Set the seed
set.seed(13)
# Initialise the network again
nn <- netup(c(4, 8, 7, 3))
# Train the network
nn <- train(nn, inp = training_inp, k = training_classes_numeric)

# Compute post-training loss using training data
post_training_loss <- loss(nn, training_inp, training_classes_numeric)
cat('Post-training loss:', post_training_loss)

## Predicting the species for the test data using the trained network

# Initialise vector to store prediction values
predictions <- rep(0, nrow(test_data))
# Iterate over each input in the test data
for (i in 1:nrow(test_inp)){
  # Use the trained neural network with the test input
  nn_output <- forward(nn, test_inp[i ,])
  # Classify the input according to the class predicted as most probable, i.e.
  # the node with the highest value in the final layer
  predictions[i] <- which.max(nn_output$h[[length(nn_output$h)]])
}

## Evaluating performance of network using misclassification rate

# Number of test inputs that were misclassified
number_misclassified <- sum(predictions != test_classes_numeric)
# Compute the misclassifcation rate as the proportion misclassfied for the test 
# set
misclassification_rate <- number_misclassified / length(predictions)
cat('Misclassification rate of the network:', round(misclassification_rate, 2))

# The misclassification rate of 0 suggests that the trained network generalises 
# well to the test data and has accurate predictions. Training the network using 
# varying seeds resulted in misclassification rates between 0-0.06, further
# supporting that the network performs well and generalises to the test data.
# Michelle Cleary (s1979093), Murad Magdiyev (s2533467), Liz Howell (s1523887)
# Contributions:
# Michelle - 
# Murad - 
# Liz - 


# The following code will 


#### Function definitions

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
  
  # Initialise empty lists to store h, W, and b
  h <- list()
  W <- list()
  b <- list()
  
  # Define the total number of layers, L
  L <- length(d)
  
  # Iterate over each layer
  for (l in 1:L) {
    # Create the nodes in layer h[[l]], initialising the elements with 
    h[[l]] <- rep(1, d[l])
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
             "b" = b)  
  return (nn)
}


forward <- function(nn, inp){
  # Function to update the network list nn by computing the remaining node 
  # values implied by inp, the given input data to be used as the values for 
  # the first layer nodes.
  #
  # Inputs:
  # nn: network list as returned by netup() containing the following elements:
  #     h: list of node values in each layer of the network
  #     W: list of corresponding weight matrices to link each layer to the next 
  #        one 
  #     b: list of offset vectors linking each layer to the next one
  # inp: vector of input values for the first layer of the network
  #
  # Outputs:
  # updated_nn: updated network list, containing the following elements: 
  #             h: list of updated node values in each layer of the network
  #             W: list of corresponding weight matrices to link each layer to 
  #                the next one (as in input nn)
  #             b: list of offset vectors linking each layer to the next one (as
  #                in input nn)
  
  # Extract h, W, and W from the input network list
  h <- nn$h
  W <- nn$W
  b <- nn$b
  
  # Set the values of the first layer nodes to be the values of the input data
  h[[1]] <- inp
  
  # Define the total number of layers, L
  L <- length(h)
  
  # Iterate over layers 1 to L - 1
  for (i in 1:(L - 1)) {
    # Compute the node values of the next layer
    h[[i+1]] <- pmax(0, W[[i]] %*% h[[i]] + b[[i]])
  }
  
  # Update the network list
  updated_nn <- list("h" = h,
                     "W" = W,
                     "b" = b)
  
  return (updated_nn)
}


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
  # k: output class
  #
  # Outputs:
  # updated_nn: updated network list containing the following elements:
  #             h: list of node values for each layer
  #             W: list of weight matrices
  #             b: list of offset vectors
  #             dh: derivatives with respect to the nodes
  #             dW: derivatives with respect to the weights
  #             db: derivatives with respect to the offsets
  
  # Extract h, W and b from input
  h <- nn$h
  W <- nn$W
  b <- nn$b
  
  # Number of layers in network
  L <- length(h)
  
  # Initialise list to store number of nodes per layer
  nodes_per_layer <- list()
  #Computing the number of nodes in that layer
    nodes_per_layer <- lengths(h)
  
  # Initialise vector d and the list dh to store derivatives w.r.t nodes for 
  # each layer
  dh <- d <- list()
  for (l in 1:L) {
    dh[[l]] <- d[[l]] <- rep(0, nodes_per_layer[l])
  }
  
  # Compute the derivative of the loss for k with respect to the nodes in final 
  # layer, L
  dh[[L]] <- exp(h[[L]]) / sum(exp(h[[L]]))
  # Assign appropriate value to node for class k
  dh[[L]][k] <- dh[[L]][k] - 1
  
  # Compute values of d for the final layer
  d[[L]] <- ifelse(h[[L]] > 0, dh[[L]], 0)
 
  # Compute the derivatives of the loss with respect to the nodes in all other
  # layers using back-propagation
  
  # Iterate backwards over each layer, starting at 2nd last layer
  for (l in (L - 1):1){
    # Compute derivative of loss with respect to the current layer as 
    # (W^l)^T d^(l + 1)
    dh[[l]] <- t(W[[l]]) %*% d[[l + 1]]
    # Compute d^l
    d[[l]] <- ifelse(h[[l]] > 0, dh[[l]], 0)
  }
  
  # Initialise lists to store derivatives with respect to offset vectors and 
  # weights for layers 1 to L - 1
  db <- list()
  dW <- list()
  for (l in 1:(L - 1)){
    db[[l]] <- rep(0, nodes_per_layer[l + 1])
    dW[[l]] <- matrix(0, nrow = nrow(W[[l]]), ncol = ncol(W[[l]]))
  }
  
  # Iterate over layers L-1 to 1
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
                     'dh' = dh,
                     'dW' = dW,
                     'db' = db)
  
  return (updated_nn)
}


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
  
  # Extract h, W, and b from input
  h <- nn$h
  W <- nn$W
  b <- nn$b
  
  # Number of layers in network
  L <- length(h)
  
  # Compute number of nodes per layer
  
  nodes_per_layer <- lengths(h)
  shifted_nodes_per_layer<-tail(nodes_per_layer,-1)
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

    all_db <- mapply(rep,0,nodes_per_layer[-1])
    all_dW<-mapply(matrix,0,shifted_nodes_per_layer,head(nodes_per_layer,-1))
    
    # Iterate over all mb sampled rows
    for (samp in 1:nrow(sampled_inp)){
      
      # Update network using forward and then back-propagation
      nn <- forward(nn, sampled_inp[samp, ])
      back_prop_gradients <- backward(nn, k = sample_class[samp])
      
        # Sum the current gradient values to the overall gradients
        all_dW <- Map("+",all_dW, back_prop_gradients$dW)
        all_db <- Map("+",all_db, back_prop_gradients$db)
    }
    
      # Update W and b using the average of the computed gradients
      nn$b <- mapply("+",nn$b,lapply(all_db,"*",(-eta/mb)))
      nn$W <- mapply("+",nn$W,lapply(all_dW,"*",(-eta/mb)))
    
  }
  
  # Return the trained network
  return (nn)
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
# training data except the class labels
# Convert to a matrix and unname the columns for use in functions
training_inp <- unname(as.matrix(training_data[, 1:4]))
test_inp <- unname(as.matrix(test_data[, 1:4]))
# Define the classes of the input for both training and testing
# Convert to a vector and unname the column 
training_classes <- unname(as.vector(training_data$Species))
test_classes <- unname(as.vector(test_data$Species))
# Convert string class names to numeric values for use in functions
training_classes_numeric <- as.numeric(factor(training_classes))
test_classes_numeric <- as.numeric(factor(test_classes))

## Training the network

# Set the seed
set.seed(7)
# Initialise the network
nn <- netup(c(4, 8, 7, 3))
# Train the network
nn <- train(nn, inp = training_inp, k = training_classes_numeric)

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


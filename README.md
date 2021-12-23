# a4
Skeleton code for Assignment 4

# Part 1: K-Nearest Neighbors Classification

## Problem Statement: 
Implement the KNN classification algorithm from scratch.

## Approach:
1. KNN classification Algorithm calculates the distances (Manhattan/Euclidean) from each point in test set to all points in train set. As per the user-defined values of "K", we pick the nearest K points(shortest distance). As per the user-defined weights parameter, we pick the mode of the target classification value of these K points to the test data point when weights parameter is "uniform". When weights paramter is "distance", we calculate the sum of the inverse of the distance of each of K data points and sum it for each target classification value. We pick the target classifier value that has the highest sum of inverse distances. 
2. I intially worked on returning the euclidean(includes diagonally) and manhattan distances(doesn't include diagonal distance) between two numpy arrays(vectors).
3. In the fit function, we just assign train and test datasets to X and y.
4. In the predict function, we calculate the distance using manhattan and eucliedean distance functions. We pick the y values as per point 1 using uniform and distance functions.
5. The accuracy of the sklearn knn library is compared with the knn algorithm built from scratch using above steps.

## Difficulties:
1. This is not a difficulty perse, but I was little confused on how to classify the test dataset using distance weights function. After checking Inscribe, I was clear and implemented it.



# Part 2: Multilayer Perceptron Classification

## Problem Statement: 
Implement the Neural Network with one hidden layer in between input and output layers from scratch.

## Approach:
1. Initially, I implemented the hidden activation functions (relu, identity, sigmoid, and tanh). Since, we will be using the derivatives of these functions, I included the derivative part in the same function with additional parameter indicating whether I want the derivative or not.
2. Secondly, I implemented the cross entropy function by taking the negative logarithmic value of predicted values and multipled it with true value.
3. I implemented the initialize function which initializes the X(predictors), y(target), hidden layer weights, outer layer weights, hidden layer bias, and outer layer bias. Hidden layer and outer layer weights are initialized in such a way that we pick random values between 0 and 1. Bias values of hidden and outer layer are initialized as ones.
4. After the initialization, I implemented the fit function. The fit function implements the forward propagation (hidden layer with hidden activation function and outer layer softmax activation function) and the backward propagation( update the delta values based on error values and derivatives of the hidden activation function). After the delta value calcuations, I updated the weights and bias values using delta and learning rates. We iterate the values of heights and bias based on the number of iterations.
5. In the predict function, I just implemented the forward propogation using the updated heights and bias values. I return the final target classifier based on the highest probability of each class.
7. The accuracy of the sklearn knn library is compared with the knn algorithm built from scratch using above steps.

## Difficulties:
1. I took some time to understand the backward propogation. After implementing the backward propagation, I got really low accuracy values. After debugging, I realized that I'm not using softmax activation function in the outer layer. After changing this, the results were better.


#%% Code needs
# 1. Needs to take in the MNIST data and seperate it into dev data and training data
# 2. Needs to initialise the structure of the network with random values for weights and biases
# 3. Needs activation functions for forward propagation with there derivatives for back propagation
# 4. Needs a forward propagation algorithm 
# 5. Needs a cost function for the back propagation algorithm
# 6. Needs a back propagation algorithm
# 7. Needs to update weights and biases based on a learing rate
# 8. Needs to run as one so that it loops over all training data

# a. output the acuracy so we can measure if the network is working as designed
# b. be able to handle single examples for demo purposes.
# c. use various activation functions for comparison.
# d. visualise 

#%% Packages
import numpy as np  # Main mathematics package
import pandas as pd  # Used for reading the data into python
import matplotlib.pyplot as plt  # Used for plotting findings

#%% Data manipulation
# My MacBook
data = pd.read_csv('/Users/joe/Library/CloudStorage/OneDrive-UniversityofPlymouth/3rd Year/Project/Neural Network/mnist_test.csv')  # Reads the csv file and stores it as a variable
# University Computers
#data = pd.read_csv('C:/Users/jadavies2/OneDrive - University of Plymouth/3rd Year/Project/Neural Network/mnist_test.csv')
data = np.array(data)  # Converts the csv data into a numpy array
m, n = data.shape  # Stores the two dimensions of the array to two variables
np.random.shuffle(data)  # Shuffles the data so the ordering is random

data_dev = data[0:5000].T  # Reserve 1000 samples for development purposes
dev_labels = data_dev[0]  # Creates an array of the labels of the images
dev_images = data_dev[1:n]  # Creates an array of the image codes
dev_images = dev_images / 255  # Normalizes image data to be between 0 and 1

train_data = data[5000:m].T  # Reserve the rest of the samples for training purposes
train_labels = train_data[0]  # Creates an array of the labels of the images
train_images = train_data[1:n]  # Creates an array of the image codes
train_images = train_images / 255  # Normalizes image data to be between 0 and 1
_, m_train = train_images.shape  # Stores the amount of training samples into a variable

#%% Common functions
def init_params():
# =============================================================================
#   Initializes a structure for the network to follow where the the first 
#   hidden layer has 16 neurons, the second hiden layer has 16 neurons and
#   the output layer has 10 neurons.
# =============================================================================
    W1 = np.random.rand(16, 784) - 0.5  # 16 by 784 matrix for the weights between the input layer and the first hidden layer
    b1 = np.random.rand(16, 1) - 0.5  # 16 by 1 matrix for the biases of the first hidden layer
    W2 = np.random.rand(16, 16) - 0.5  # 16 by 16 matrix for the weights between the first and second hiden layer
    b2 = np.random.rand(16, 1) - 0.5  # 16 by 1 matrix for the biases of the second hidden layer
    W3 = np.random.rand(10, 16) - 0.5  # 10 by 16 matrix for the weights between the second hiden layer and the output layer
    b3 = np.random.rand(10, 1) - 0.5  # 10 by 1 matrix for the biases of the output layer
    return W1, b1, W2, b2, W3, b3  # Outputs the matrices above

def softmax(Z):
# =============================================================================
#   Given a vector Z, applys the softmax function softmax(Z) (Vector).
#   Parameters:
#   Z = The vector for the output layer.
# =============================================================================
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def cost_exp(Y):
# =============================================================================
#   Creates a matrix comprsed of the vectors for the expected value used
#   in the cost function in back propagation.
#   Parameters:
#   Y = Vector of all the labels for the image data.
# =============================================================================
    cost_exp_Y = np.zeros((Y.size, Y.max() + 1))  # Creates a matrix of zeros of size (training examples) x (amount of classes)
    cost_exp_Y[np.arange(Y.size), Y] = 1  # Puts a 1 in the position of the row that coresponds to the label
    cost_exp_Y = cost_exp_Y.T  # Transposes the above matrix
    return cost_exp_Y

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
# =============================================================================
#   Performs the adjustments for for the weights and biases in the network 
#   based on a given learning rate (alpha).
#   Parameters:
#   Wi = Existing weights in the network for the ith layer.
#   bi = Existing biases in the network for the ith layer.
#   dWi = Adjustments to make to the ith layers weights.
#   dbi = Adjustments to make to the ith layers biases.
#   alpha = The learning rate for the training process
# =============================================================================
    W1 = W1 - alpha * dW1  # Adjustment to W1
    b1 = b1 - alpha * db1  # Adjustment to b1
    W2 = W2 - alpha * dW2  # Adjustment to W2
    b2 = b2 - alpha * db2  # Adjustment to b2
    W3 = W3 - alpha * dW3  # Adjustment to W3
    b3 = b3 - alpha * db3  # Adjustment to b3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3):
# =============================================================================
#   Returns the class of the most activated neuron in the output layer.
# =============================================================================
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
# =============================================================================
#   Prints two vectors, one of the predictions that the network has made and
#   one of the actual labels for the given examples. Then returns the 
#   percentage of predictions that are correct.
#   Parameters:
#   predictions = Vector of predictions.
#   Y = The vector of the actual labels
# =============================================================================
    print(predictions, Y)  # Prints the two vectors
    return np.sum(predictions == Y) / Y.size  # Outputs the proportion of correct classifications

#%% ReLU functions
def ReLU(Z):
# =============================================================================
#   Given a vector Z, outputs the activation function ReLU(Z) (Vector).
#   Parameters:
#   Z = The vector of the layer before the activation function is applied.
# =============================================================================
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
# =============================================================================
#   Derivative of the ReLU function used in the back propagation algorithm.
#   Parameters:
#   Z = The vector of the layer before the activation function is applied.
# =============================================================================
    return Z > 0

def forward_prop_ReLU(W1, b1, W2, b2, W3, b3, X):
# =============================================================================
#   Runs through form the input layer to the output layer with the forward
#   propagation algorithm.
#   Parameters:
#   Wi = Weights of the ith layer.
#   bi = Biases of the ith layer.
#   X = Matrix of the image data that the network uses.
# =============================================================================
    Z1 = W1.dot(X) + b1  # One step of the forward propagation algorithm without the activation function
    A1 = ReLU(Z1)  # ReLU of the previous step
    Z2 = W2.dot(A1) + b2  # Next step wo/ activation function
    A2 = ReLU(Z2)  # Activation function
    Z3 = W3.dot(A2) + b3  # Next step wo/ activation function
    A3 = softmax(Z3)  # Softmax function of the previous step for better classification
    return Z1, A1, Z2, A2, Z3, A3  # Outputs the previous vectors

def backward_prop_ReLU(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
# =============================================================================
#   Runs through the backpropagation algorithm to create a series of vectors 
#   of the proportional adjustments to make to the weights and biases.
#   Parameters:
#   Zi = Vectors of the ith layer without the activation function.
#   Ai = Vectors of the ith layer with the activation function.
#   X = Inputed image data matrix.
#   Y = Inputed image label vector.
#   m = Amount of examples used
# =============================================================================
    cost_exp_Y = cost_exp(Y)  # Stores the cost expected values matrix
    
    dZ3 = A3 - cost_exp_Y  # Calculates the cost of network
    dW3 = 1 / m * dZ3.dot(A2.T)  # Matrix of weight adjustments for W3
    db3 = 1 / m * np.sum(dZ3)  # Matrix of bias adjustments for b3
    
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2) 
    dW2 = 1 / m * dZ2.dot(A1.T)  # Matrix of weight adjustments for W2
    db2 = 1 / m * np.sum(dZ2)  # Matrix of bias adjustments for b2
    
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)  # Matrix of weight adjustments for W1
    db1 = 1 / m * np.sum(dZ1)  # Matrix of bias adjustments for b1
    return dW1, db1, dW2, db2, dW3, db3

def gradient_descent_ReLU(X, Y, alpha, iterations):
# =============================================================================
#   Puts together the above functions so they can work together as a working
#   network. Trains the network on a given number of examples / itterations
#   with a given learning rate.
#   Parameters:
#   X = Matrix of image data.
#   Y = Vector of image labels.
#   alpha = The learning rate of the network
#   itterations = The number of examples / itterations the network leawrns from.
# =============================================================================
    W1, b1, W2, b2, W3, b3 = init_params()  # Creates the structure of the network with random weights and biases
    accuracies = []  # Creates an emply list for the accuracies to be stored in
    for i in range(iterations):  # Starts a (itterations) long loop to run the learning methods
        Z1, A1, Z2, A2, Z3, A3 = forward_prop_ReLU(W1, b1, W2, b2, W3, b3, X)  # Forward propagation
        dW1, db1, dW2, db2, dW3, db3 = backward_prop_ReLU(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)  # Backwards propagation
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)  # Updating of weights and biases
        if i % 10 == 0:  # Runs a sub loop for every 10 itterations
            predictions = get_predictions(A3)  # Calculates the predictions
            accuracy = get_accuracy(predictions, Y)  # Calculates the accuracy
            accuracies.append(accuracy)  # Stores the acuracy in the accuracies list
            print(f"(ReLU) Iteration: {i}, Accuracy: {get_accuracy(predictions, Y):.4f}")  # Prints what itteration the loop is on and what the acuracy is for that stage in the loop
    return W1, b1, W2, b2, W3, b3, accuracies

def make_predictions_ReLU(X, W1, b1, W2, b2, W3, b3):
# =============================================================================
#   Inputs training examples to the network so it can output there 
#   classification.
#   Parameters:
#   x = Matrix of image data.
#   Wi = Matrix of weight for the ith layer.
#   bi = Vector of all the biases for the ith layer.
# =============================================================================
    _, _, _, _, _, A3 = forward_prop_ReLU(W1, b1, W2, b2, W3, b3, X)  # Puts image data through the network without running back propagation to see if the set weights and biases are effective
    predictions = get_predictions(A3)  # Uses the get_predictions function to classify the proseced image data
    return predictions

def test_prediction_ReLU(index, W1, b1, W2, b2, W3, b3):
# =============================================================================
#   Predicts thae sclassification of a specific example and then plots the 
#   image alog with its label and its prediction.
#   Parameters:
#   index = The index for the specific example you want to plot.
#   Wi = Matrix of weights for the ith layer.
#   bi = Vector of biases for the ith layer.
# =============================================================================
    current_image = train_images[:, index, None]  # Singles out the image data vector from the image data matrix from index
    prediction = make_predictions_ReLU(train_images[:, index, None], W1, b1, W2, b2, W3, b3)  # Makes the prediction for the singled out image data
    prediction = int(prediction)
    label = train_labels[index]  # Stores the correct label for this example into th e variable (label)
    
    current_image = current_image.reshape((28, 28)) * 255  # Reshapes the 784 entry vector of pixel data and reshapes it into a 28 * 28 square as the image was designed
    plt.gray()  # Plotting in gray scale
    plt.imshow(current_image, interpolation='nearest')  # Plots the image 
    plt.title(f'(ReLU) Label = {label}, Prediction = {prediction}')  # Puts the actual label and the estimate in the title
    plt.show()  # Shows the full plot
    
#%% Tanh functions
def tanh(Z):
# =============================================================================
#   Given a vector Z, outputs the activation function tanh(Z) (Vector).
#   Parameters:
#   Z = The vector of the layer before the activation function is applied.
# =============================================================================
    return np.tanh(Z)

def tanh_deriv(Z):
# =============================================================================
#   Derivative of the tanh function used in the back propagation algorithm.
#   Parameters:
#   Z = The vector of the layer before the activation function is applied.
# =============================================================================
    return 1 - (np.tanh(Z))**2

def forward_prop_tanh(W1, b1, W2, b2, W3, b3, X):
# =============================================================================
#   Runs through form the input layer to the output layer with the forward
#   propagation algorithm.
#   Parameters:
#   Wi = Weights of the ith layer.
#   bi = Biases of the ith layer.
#   X = Matrix of the image data that the network uses.
# =============================================================================
    Z1 = W1.dot(X) + b1  # One step of the forward propagation algorithm without the activation function
    A1 = tanh(Z1)  # tanh of the previous step
    Z2 = W2.dot(A1) + b2  # Next step wo/ activation function
    A2 = tanh(Z2)  # Activation function
    Z3 = W3.dot(A2) + b3  # Next step wo/ activation function
    A3 = softmax(Z3)  # Softmax function of the previous step for better classification
    return Z1, A1, Z2, A2, Z3, A3  # Outputs the previous vectors

def backward_prop_tanh(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
# =============================================================================
#   Runs through the backpropagation algorithm to create a series of vectors 
#   of the proportional adjustments to make to the weights and biases.
#   Parameters:
#   Zi = Vectors of the ith layer without the activation function.
#   Ai = Vectors of the ith layer with the activation function.
#   X = Inputed image data matrix.
#   Y = Inputed image label vector.
#   m = Amount of examples used
# =============================================================================
    cost_exp_Y = cost_exp(Y)  # Stores the cost expected values matrix
    
    dZ3 = A3 - cost_exp_Y  # Calculates the cost of network
    dW3 = 1 / m * dZ3.dot(A2.T)  # Matrix of weight adjustments for W3
    db3 = 1 / m * np.sum(dZ3)  # Matrix of bias adjustments for b3
    
    dZ2 = W3.T.dot(dZ3) * tanh_deriv(Z2) 
    dW2 = 1 / m * dZ2.dot(A1.T)  # Matrix of weight adjustments for W2
    db2 = 1 / m * np.sum(dZ2)  # Matrix of bias adjustments for b2
    
    dZ1 = W2.T.dot(dZ2) * tanh_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)  # Matrix of weight adjustments for W1
    db1 = 1 / m * np.sum(dZ1)  # Matrix of bias adjustments for b1
    return dW1, db1, dW2, db2, dW3, db3

def gradient_descent_tanh(X, Y, alpha, iterations):
# =============================================================================
#   Puts together the above functions so they can work together as a working
#   network. Trains the network on a given number of examples / itterations
#   with a given learning rate.
#   Parameters:
#   X = Matrix of image data.
#   Y = Vector of image labels.
#   alpha = The learning rate of the network
#   itterations = The number of examples / itterations the network leawrns from.
# =============================================================================
    W1, b1, W2, b2, W3, b3 = init_params()  # Creates the structure of the network with random weights and biases
    accuracies = []  # Creates an emply list for the accuracies to be stored in
    for i in range(iterations):  # Starts a (itterations) long loop to run the learning methods
        Z1, A1, Z2, A2, Z3, A3 = forward_prop_tanh(W1, b1, W2, b2, W3, b3, X)  # Forward propagation
        dW1, db1, dW2, db2, dW3, db3 = backward_prop_tanh(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)  # Backwards propagation
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)  # Updating of weights and biases
        if i % 10 == 0:  # Runs a sub loop for every 10 itterations
            predictions = get_predictions(A3)  # Calculates the predictions
            accuracy = get_accuracy(predictions, Y)  # Calculates the accuracy
            accuracies.append(accuracy)  # Stores the acuracy in the accuracies list
            print(f"(tanh) Iteration: {i}, Accuracy: {get_accuracy(predictions, Y):.4f}")  # Prints what itteration the loop is on and what the acuracy is for that stage in the loop
    return W1, b1, W2, b2, W3, b3, accuracies

def make_predictions_tanh(X, W1, b1, W2, b2, W3, b3):
# =============================================================================
#   Inputs training examples to the network so it can output there 
#   classification.
#   Parameters:
#   x = Matrix of image data.
#   Wi = Matrix of weight for the ith layer.
#   bi = Vector of all the biases for the ith layer.
# =============================================================================
    _, _, _, _, _, A3 = forward_prop_tanh(W1, b1, W2, b2, W3, b3, X)  # Puts image data through the network without running back propagation to see if the set weights and biases are effective
    predictions = get_predictions(A3)  # Uses the get_predictions function to classify the proseced image data
    return predictions

def test_prediction_tanh(index, W1, b1, W2, b2, W3, b3):
# =============================================================================
#   Predicts thae sclassification of a specific example and then plots the 
#   image alog with its label and its prediction.
#   Parameters:
#   index = The index for the specific example you want to plot.
#   Wi = Matrix of weights for the ith layer.
#   bi = Vector of biases for the ith layer.
# =============================================================================
    current_image = train_images[:, index, None]  # Singles out the image data vector from the image data matrix from index
    prediction = make_predictions_tanh(train_images[:, index, None], W1, b1, W2, b2, W3, b3)  # Makes the prediction for the singled out image data
    prediction = int(prediction)
    label = train_labels[index]  # Stores the correct label for this example into th e variable (label)
    
    current_image = current_image.reshape((28, 28)) * 255  # Reshapes the 784 entry vector of pixel data and reshapes it into a 28 * 28 square as the image was designed
    plt.gray()  # Plotting in gray scale
    plt.imshow(current_image, interpolation='nearest')  # Plots the image 
    plt.title(f'(tanh) Label = {label}, Prediction = {prediction}')  # Puts the actual label and the estimate in the title
    plt.show()  # Shows the full plot
    
#%% Sigmoid functions
def init_params_sigmoid():
# =============================================================================
#   Initializes a structure for the network to follow where the the first 
#   hidden layer has 16 neurons, the second hiden layer has 16 neurons and
#   the output layer has 10 neurons.
# =============================================================================
    W1 = np.random.rand(16, 784)  # 16 by 784 matrix for the weights between the input layer and the first hidden layer
    b1 = np.random.rand(16, 1)  # 16 by 1 matrix for the biases of the first hidden layer
    W2 = np.random.rand(16, 16)  # 16 by 16 matrix for the weights between the first and second hiden layer
    b2 = np.random.rand(16, 1)  # 16 by 1 matrix for the biases of the second hidden layer
    W3 = np.random.rand(10, 16)  # 10 by 16 matrix for the weights between the second hiden layer and the output layer
    b3 = np.random.rand(10, 1)  # 10 by 1 matrix for the biases of the output layer
    return W1, b1, W2, b2, W3, b3  # Outputs the matrices above
    
def sigmoid(Z):
# =============================================================================
#   Given a vector Z, outputs the activation function sigmoid(Z) (Vector).
#   Parameters:
#   Z = The vector of the layer before the activation function is applied.
# =============================================================================
    return np.exp(Z) / (1 + np.exp(Z))

def sigmoid_deriv(Z):
# =============================================================================
#   Derivative of the tanh function used in the back propagation algorithm.
#   Parameters:
#   Z = The vector of the layer before the activation function is applied.
# =============================================================================
    return np.exp(Z) / (1 + np.exp(Z))**2

def forward_prop_sigmoid(W1, b1, W2, b2, W3, b3, X):
# =============================================================================
#   Runs through form the input layer to the output layer with the forward
#   propagation algorithm.
#   Parameters:
#   Wi = Weights of the ith layer.
#   bi = Biases of the ith layer.
#   X = Matrix of the image data that the network uses.
# =============================================================================
    Z1 = W1.dot(X) + b1  # One step of the forward propagation algorithm without the activation function
    A1 = sigmoid(Z1)  # tanh of the previous step
    Z2 = W2.dot(A1) + b2  # Next step wo/ activation function
    A2 = sigmoid(Z2)  # Activation function
    Z3 = W3.dot(A2) + b3  # Next step wo/ activation function
    A3 = softmax(Z3)  # Softmax function of the previous step for better classification
    return Z1, A1, Z2, A2, Z3, A3  # Outputs the previous vectors

def backward_prop_sigmoid(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
# =============================================================================
#   Runs through the backpropagation algorithm to create a series of vectors 
#   of the proportional adjustments to make to the weights and biases.
#   Parameters:
#   Zi = Vectors of the ith layer without the activation function.
#   Ai = Vectors of the ith layer with the activation function.
#   X = Inputed image data matrix.
#   Y = Inputed image label vector.
#   m = Amount of examples used
# =============================================================================
    cost_exp_Y = cost_exp(Y)  # Stores the cost expected values matrix
    
    dZ3 = A3 - cost_exp_Y  # Calculates the cost of network
    dW3 = 1 / m * dZ3.dot(A2.T)  # Matrix of weight adjustments for W3
    db3 = 1 / m * np.sum(dZ3)  # Matrix of bias adjustments for b3
    
    dZ2 = W3.T.dot(dZ3) * sigmoid_deriv(Z2) 
    dW2 = 1 / m * dZ2.dot(A1.T)  # Matrix of weight adjustments for W2
    db2 = 1 / m * np.sum(dZ2)  # Matrix of bias adjustments for b2
    
    dZ1 = W2.T.dot(dZ2) * sigmoid_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)  # Matrix of weight adjustments for W1
    db1 = 1 / m * np.sum(dZ1)  # Matrix of bias adjustments for b1
    return dW1, db1, dW2, db2, dW3, db3

def gradient_descent_sigmoid(X, Y, alpha, iterations):
# =============================================================================
#   Puts together the above functions so they can work together as a working
#   network. Trains the network on a given number of examples / itterations
#   with a given learning rate.
#   Parameters:
#   X = Matrix of image data.
#   Y = Vector of image labels.
#   alpha = The learning rate of the network
#   itterations = The number of examples / itterations the network leawrns from.
# =============================================================================
    W1, b1, W2, b2, W3, b3 = init_params()  # Creates the structure of the network with random weights and biases
    accuracies = []  # Creates an emply list for the accuracies to be stored in
    for i in range(iterations):  # Starts a (itterations) long loop to run the learning methods
        Z1, A1, Z2, A2, Z3, A3 = forward_prop_sigmoid(W1, b1, W2, b2, W3, b3, X)  # Forward propagation
        dW1, db1, dW2, db2, dW3, db3 = backward_prop_sigmoid(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)  # Backwards propagation
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)  # Updating of weights and biases
        if i % 10 == 0:  # Runs a sub loop for every 10 itterations
            predictions = get_predictions(A3)  # Calculates the predictions
            accuracy = get_accuracy(predictions, Y)  # Calculates the accuracy
            accuracies.append(accuracy)  # Stores the acuracy in the accuracies list
            print(f"(sigmoid) Iteration: {i}, Accuracy: {get_accuracy(predictions, Y):.4f}")  # Prints what itteration the loop is on and what the acuracy is for that stage in the loop
    return W1, b1, W2, b2, W3, b3, accuracies

def make_predictions_sigmoid(X, W1, b1, W2, b2, W3, b3):
# =============================================================================
#   Inputs training examples to the network so it can output there 
#   classification.
#   Parameters:
#   x = Matrix of image data.
#   Wi = Matrix of weight for the ith layer.
#   bi = Vector of all the biases for the ith layer.
# =============================================================================
    _, _, _, _, _, A3 = forward_prop_sigmoid(W1, b1, W2, b2, W3, b3, X)  # Puts image data through the network without running back propagation to see if the set weights and biases are effective
    predictions = get_predictions(A3)  # Uses the get_predictions function to classify the proseced image data
    return predictions

def test_prediction_sigmoid(index, W1, b1, W2, b2, W3, b3):
# =============================================================================
#   Predicts thae sclassification of a specific example and then plots the 
#   image alog with its label and its prediction.
#   Parameters:
#   index = The index for the specific example you want to plot.
#   Wi = Matrix of weights for the ith layer.
#   bi = Vector of biases for the ith layer.
# =============================================================================
    current_image = train_images[:, index, None]  # Singles out the image data vector from the image data matrix from index
    prediction = make_predictions_sigmoid(train_images[:, index, None], W1, b1, W2, b2, W3, b3)  # Makes the prediction for the singled out image data
    prediction = int(prediction)
    label = train_labels[index]  # Stores the correct label for this example into th e variable (label)
    
    current_image = current_image.reshape((28, 28)) * 255  # Reshapes the 784 entry vector of pixel data and reshapes it into a 28 * 28 square as the image was designed
    plt.gray()  # Plotting in gray scale
    plt.imshow(current_image, interpolation='nearest')  # Plots the image 
    #plt.title(f'(sigmoid) Label = {label}, Prediction = {prediction}')  # Puts the actual label and the estimate in the title
    plt.axis('off')
    plt.show()  # Shows the full plot
    
    return current_image
#%% Visualisation functions
def visualise_weights(W1):
 
    num_neurons = W1.shape[0]
    plt.figure(figsize=(10, 10))

    for i in range(num_neurons):
        weight_image = W1[i].reshape(28, 28)
        plt.subplot(4, 4, i+1)
        plt.imshow(weight_image, cmap='gray')
        plt.title(f"Neuron {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
#%% Training models
alpha = 0.2

W1_ReLU, b1_ReLU, W2_ReLU, b2_ReLU, W3_ReLU, b3_ReLU, accuracies_ReLU = gradient_descent_ReLU(dev_images, dev_labels, alpha, 5000)
W1_tanh, b1_tanh, W2_tanh, b2_tanh, W3_tanh, b3_tanh, accuracies_tanh = gradient_descent_tanh(dev_images, dev_labels, alpha, 5000)
W1_sigmoid, b1_sigmoid, W2_sigmoid, b2_sigmoid, W3_sigmoid, b3_sigmoid, accuracies_sigmoid = gradient_descent_sigmoid(dev_images, dev_labels, alpha, 5000)

#%% Plotting of accuracy over itterations for ReLU
plt.plot(accuracies_ReLU, label = 'ReLU')
plt.plot(accuracies_tanh, label = 'tanh')
plt.plot(accuracies_sigmoid, label = 'sigmoid')
plt.xlim(0,300)
plt.xlabel('Itterations, x10')
plt.ylabel('Accuracy, x100%')
plt.title(f'Training Accuracy Over Iterations for alpha = {alpha}')
plt.grid()
plt.legend()
plt.show()

#%% Varring alpha 
alpha = 0.1
W1_1, b1_1, W2_1, b2_1, W3_1, b3_1, accuracies_1 = gradient_descent_ReLU(dev_images, dev_labels, alpha, 1000)

alpha = 0.2
W1_2, b1_2, W2_2, b2_2, W3_2, b3_2, accuracies_2 = gradient_descent_ReLU(dev_images, dev_labels, alpha, 1000)

alpha = 0.3
W1_3, b1_3, W2_3, b2_3, W3_3, b3_3, accuracies_3 = gradient_descent_ReLU(dev_images, dev_labels, alpha, 1000)

alpha = 0.4
W1_4, b1_4, W2_4, b2_4, W3_4, b3_4, accuracies_4 = gradient_descent_ReLU(dev_images, dev_labels, alpha, 1000)

alpha = 0.5
W1_5, b1_5, W2_5, b2_5, W3_5, b3_5, accuracies_5 = gradient_descent_ReLU(dev_images, dev_labels, alpha, 1000)

alpha = 0.6
W1_6, b1_6, W2_6, b2_6, W3_6, b3_6, accuracies_6 = gradient_descent_ReLU(dev_images, dev_labels, alpha, 1000)

alpha = 0.7
W1_7, b1_7, W2_7, b2_7, W3_7, b3_7, accuracies_7 = gradient_descent_ReLU(dev_images, dev_labels, alpha, 1000)

alpha = 0.8
W1_8, b1_8, W2_8, b2_8, W3_8, b3_8, accuracies_8 = gradient_descent_ReLU(dev_images, dev_labels, alpha, 1000)

alpha = 0.9
W1_9, b1_9, W2_9, b2_9, W3_9, b3_9, accuracies_9 = gradient_descent_ReLU(dev_images, dev_labels, alpha, 1000)

#%%
plt.plot(accuracies_1, label = 'alpha = 0.1')
#plt.plot(accuracies_2, label = 'alpha = 0.2')
#plt.plot(accuracies_3, label = 'alpha = 0.3')
plt.plot(accuracies_4, label = 'alpha = 0.4')
#plt.plot(accuracies_5, label = 'alpha = 0.5')
#plt.plot(accuracies_6, label = 'alpha = 0.6')
#plt.plot(accuracies_7, label = 'alpha = 0.7')
#plt.plot(accuracies_8, label = 'alpha = 0.8')
plt.plot(accuracies_9, label = 'alpha = 0.9')
plt.xlabel('Itterations, x10')
plt.ylabel('Accuracy, x100%')
plt.title(f'Training Accuracy Over Iterations for varring alpha')
plt.grid()
plt.legend()
plt.show()

#%% Unique examples
for i in range(0, 25):
    test_prediction_ReLU(i, W1_ReLU, b1_ReLU, W2_ReLU, b2_ReLU, W3_ReLU, b3_ReLU)
    test_prediction_tanh(i, W1_tanh, b1_tanh, W2_tanh, b2_tanh, W3_tanh, b3_tanh)
    test_prediction_sigmoid(i, W1_sigmoid, b1_sigmoid, W2_sigmoid, b2_sigmoid, W3_sigmoid, b3_sigmoid)

#%% Weight visualisations
visualise_weights(W1_ReLU)
visualise_weights(W1_tanh)
visualise_weights(W1_sigmoid)

visualise_weights(W1_1)
visualise_weights(W1_2)
visualise_weights(W1_3)
visualise_weights(W1_4)
visualise_weights(W1_5)
visualise_weights(W1_6)
visualise_weights(W1_7)
visualise_weights(W1_8)
visualise_weights(W1_9)

#%% Testing
digit = test_prediction_sigmoid(8, W1_sigmoid, b1_sigmoid, W2_sigmoid, b2_sigmoid, W3_sigmoid, b3_sigmoid)

























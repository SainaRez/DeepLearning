# This code is an n layer neural network that uses the MNIST dataset to train the neural net to classify hand written digits.
# The training data is 55000 (26x28) images of handwritten data in the shape of a flattened 784 dimentional vector.
# Validation data is 5000 images and testing data includes 10000 images.
# The output data of the model includes all the images with probabilities of all digits for each instance. 
# The code uses the cross entropy loss function, forward and backward propogation. All the layers have the relu activation layer and the last layer uses softmax.
# This code is written in Python 2.7
# Training_data: 55000 x 784


import numpy as np
import math


def calc_softmax(w, activation_matrix, b):
    """  
    Inputs: The list of weights with the length of the number of hidden layers + input layer. Last activation matrix (batch size x hidden units), Biases for the output layer (10 x 1)
    Output: (output layer yhat) probabilites of the digits (batch size x 10)
    """
    z_data = np.dot(activation_matrix, w) + b
    yhat = (np.exp(z_data.T) / np.sum(np.exp(z_data), axis=1)).T
    return yhat


def relu(w, training_data, b):
    """
    Inputs: w is the list of weights with the length of the number of hidden layers + input layer. 
    Activation matrix from the layer before or input data(batch size x hidden units(size of each data sample for input)), Biases for that layer (hidden units x 1)
    Output: the activation matrix for that layer (batch size x hidden units)
    """
    z_data = np.dot(training_data, w) + b
    activation_matrix = np.maximum(z_data, 0)
    return activation_matrix

def relu_prime(layer, activation_list):
    """
    Input: The layer backprop is on, list of activation matrices
    Output: da_dz (dirivative of activation with respect to z) 
    """
    a = activation_list[layer]
    return np.where(a < 0, 0, 1)


def forward(weights, biases, training_data):
    """
    This furnction performs forward propogation by applying relu to the input layer and the hidden layers and softmax to the last or ouput layer
    Input: list of randomly initialized weights and zero initialized biases, data_batch
    Output: predictions or output layer(batch size x 10), z: aw + b, list of activation matrices (len = layers, each matrix is (batch size x hidden units))
    """
    activation_list = [training_data]
    z = [training_data]

    # activation = relu(weights[0], training_data, biases[0])
    # activation_list.append(activation)
    activation = training_data
    for i in range(len(weights)-1):
        z.append(activation.dot(weights[i])+biases[i])
        activation = relu(weights[i], activation, biases[i])
        activation_list.append(activation)
    
    z.append(activation.dot(weights[-1]) + biases[-1])
    yhat = calc_softmax(weights[len(weights)-1], activation, biases[len(biases)-1])
    activation_list.append(yhat)

    return yhat, z, activation_list



def init_weights(num_inputs, input_hidden_layers, unit_count, training_labels):
    """
    Input: number of hidden layers + input layer, number of hidden units, labels
    Outpu: A list of weights with the size of the layers where each matrix is initialized with random numbers. First weight is (size of data x hidden units), last weight is (hidden units x 10)
    and the rest of the weights are (hidden units x hidden units)
    """
    weights = []
    cons = 0.1
    first_w = cons * np.random.randn(num_inputs, unit_count)
    weights.append(first_w)
    last_w = cons * np.random.randn(unit_count, training_labels.shape[1])
    for i in range(input_hidden_layers - 2):
        w = cons * np.random.rand(unit_count, unit_count)
        weights.append(w)
    weights.append(last_w)
    return weights


def init_biases(input_hidden_layers, unit_count, training_labels):
    """
    Input: number of hidden layers + input layer, number of hidden units, labels
    Output: A list of biases with the same length as weights all initialized to zeros. Last bias is a vector of 10 and the rest of vectors of the number of hidden units
    """
    biases = []
    for _ in range(input_hidden_layers - 1): 
        biases.append(np.zeros(unit_count))

    last_b = np.zeros(training_labels.shape[1])
    biases.append(last_b)
    return biases


def backward(weights, biases, input_hidden_layers, batch_size, yhat, activation_list, input_data, labels, z):
    """
    This function initializes g for the last layer (output layer). g is the derivative of loss function with respect to yhat (predictions). It calculates dL_dw and dL_db for the out layer, z
    then it calculates them for the rest of the layers by multipllying g with the derivative of activations with respect to z. 
    Input: The initialized weights and biases, number of hidden layers + input layer, batch size, list of activation matrices, input data and labels
    Output: delta w: The list of derivatives of the loss function with respect to w (dL_dw)
    delta b: delta b: The list of derivatives of the loss function with respect to b (dL_db), learning rate
    """
    delta_w = []
    delta_b = []
    g = yhat - labels

    dL_dw = (1./batch_size) * np.dot(g.T, activation_list[-2])
    dL_db = (1./batch_size) * g.sum(axis=0)
    delta_w.append(1./batch_size * dL_dw)
    delta_b.append(dL_db)
    for i in range(-1, -len(weights), -1):
        g = np.dot(g, weights[i].T) * relu_prime(i-1, z)
        dL_dw = (1./batch_size) * np.dot(g.T, activation_list[i-2])
        dL_db = (1./batch_size) * g.sum(axis=0)
        #dL_db = np.mean(g, axis=0)
        delta_w.append(dL_dw)
        delta_b.append(dL_db)
    
    return delta_w, delta_b


def update_variables(weights, biases, delta_w, delta_b, learning_rate):
    """
    Input: List of weights, list of biases, delta w: The list of derivatives of the loss function with respect to w (dL_dw),
    delta b: The list of derivatives of the loss function with respect to b (dL_db), learning rate
    Output: The updated weights and biases
    """
    delta_w = delta_w[::-1]
    delta_b = delta_b[::-1]
    for k in range(len(weights)):
        weights[k] -= learning_rate * delta_w[k].T
    for k in range(len(biases)):
        biases[k] -= learning_rate * delta_b[k]
    
    return weights, biases


def train(training_data, training_labels, layer_count, unit_count, epsilon, batch_size):
    """
    This function devides the data into batches and runs forward and backward propogation for each batch and updates the weights and the biases. 
    It repeats this process for the number of epochs
    Input: input data, the labels, number of layers, number of hidden units and batch size
    Output: the final matrices of updated weights and biases
    """
    n, m = training_data.shape
    input_hidden_layers = layer_count - 1

    # Initialize weights and biases
    weights = init_weights(m, input_hidden_layers, unit_count, training_labels) 
    biases = init_biases(input_hidden_layers, unit_count, training_labels)

    data_size = training_data.shape[0]
    random_indexes = np.random.choice(data_size, data_size, replace=False)
    epoch = 80

    for e in range(epoch):
        if e % 20 == 0:
            print "Epoch:", e
        i = 0
        for i in range(0, n, batch_size):    # Loops every batch size
            data_batch = training_data[random_indexes[i: i+batch_size]]
            label_batch = training_labels[random_indexes[i: i+batch_size]]

            #Performing forward 
            yhat, z, activation_list = forward(weights, biases, data_batch)

            #Perfoming backward
            delta_w, delta_b = backward(weights, biases, input_hidden_layers, batch_size, yhat, activation_list, data_batch, label_batch, z)

            # Update weights and biases    
            weights, biases = update_variables(weights, biases, delta_w, delta_b, epsilon)

    return weights, biases


def findBestHyperparameters(training_data, training_labels, validation_data, validation_labels, layers, batch):
    """
    Inputs: training data, training labels, validation data, validation labels, number of layers and number of batches
    Ouput: Finds the best learning rate and number of hidden units
    """
    print "Finding the best hyper parameters ..."
    accuracy_dict = {}
    #batch_size = [128, 600, 1000]
    learning_rate = [0.1, 0.01, 0.001]
    hidden_units = [30, 50, 70]

    #for batch in batch_size:
    for rate in learning_rate:
        for unit in hidden_units:
            # First train the data
            weights, biases = train(training_data, training_labels, layers, unit, rate, batch)
            # Then find the accuracy on validation data
            acc = accuracy(weights, biases, validation_data, validation_labels)
            accuracy_dict[acc] = [rate, unit]

    # We find the parameters corresponding to the highest accuracy
    accuracy_keys = sorted(accuracy_dict.keys())
    highest_acc = accuracy_keys[-1]
    rate, unit = accuracy_dict[highest_acc]
    print "Found the best hyper parameters !!!"
    print "learning_rate:", rate, "unit:", unit
    return rate, unit




def accuracy(weights, biases, testing_data, testing_labels):
    """
    Input: Initialized weights and biases, testing data, testing labels
    Output: The calculated accuracy of predictions compared to the labels
    """
    predictions, _, _ = forward(weights, biases, testing_data)
    accuracy_counter = 0.0
    for row_index in range(len(predictions)):
        if np.argmax(predictions[row_index]) == np.argmax(testing_labels[row_index]):
            accuracy_counter += 1.0
    accuracy = (accuracy_counter/testing_labels.shape[0]) * 100
    print "accuracy:", accuracy, "%"
    return accuracy


if __name__== "__main__":

    # Loading the MNIST data
    testing_data = np.load("mnist_test_images.npy")
    testing_labels = np.load("mnist_test_labels.npy")
    training_data = np.load("mnist_train_images.npy")
    training_labels = np.load("mnist_train_labels.npy")
    validation_data = np.load("mnist_validation_images.npy")
    validation_labels = np.load("mnist_validation_labels.npy")

    batch_size = 128
    layer_count = 5
    #learning_rate = 0.1
    #unit_count = 50

    # To run this neural net faster, directly pass in the learning rate and unit_count without calling find_hyperparameter()
  
    learning_rate, unit_count = findBestHyperparameters(training_data, training_labels, validation_data, validation_labels, layer_count, batch_size)

    weights, biases = train(training_data, training_labels, layer_count, unit_count, learning_rate, batch_size)
    accuracy(weights, biases, testing_data, testing_labels)




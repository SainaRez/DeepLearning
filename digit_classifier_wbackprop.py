# This code is an n layer neural network that uses the MNIST dataset to train the neural net to classify hand written digits.
# The training data is 55000 (26x28) images of handwritten data in the shape of a flattened 784 dimentional vector.
# Validation data is 5000 images and testing data includes 10000 images.
# The output data of the model includes all the images with probabilities of all digits for each instance. 
# The code uses the cross entropy loss function, forward and backward propogation. All the layers have the relu activation layer and the last layer uses softmax.
# This code is written in Python 2.7
# Training_data: 55000 x 784


import numpy as np


def calc_softmax(w, activation_matrix, b):
    """  
    Inputs: The list of weights with the length of the number of hidden layers + input layer. Last activation matrix (number ofsamples x hidden units), Biases for the output layer (10 x 1)
    Output: (output layer yhat) probabilites of the digits (number of samples x 10)
    """
    z_data = np.dot(activation_matrix, w) + b
    yhat = (np.exp(z_data.T) / np.sum(np.exp(z_data), axis=1)).T
    return yhat


def relu(w, training_data, b):
    """
    Inputs: w is the list of weights with the length of the number of hidden layers + input layer. 
    Activation matrix from the layer before or input data(number of samples x hidden units(each sample size for input)), Biases for that layer (hidden units x 1)
    Output: the activation matrix for that layer (number of samples x hidden units)
    """
    z_data = np.dot(training_data, w) + b
    activation_matrix = np.maximum(z_data, 0)
    return activation_matrix

def relu_prime(layer, activation_list):
    a = activation_list[layer]
    return np.where(a < 0, 0, 1)


# The cross-entropy loss function 
# Inputs: wights w (784x10), input data, labels, alpha: regularization constant,  Output: j: the cost value
def loss(training_labels, yhat):
    j = training_labels * np.log(yhat)
    j = np.mean(j) * (0.5) * (-1)
    #j = j + ((alpha * 0.5) * np.sum(np.dot(w,w.T)))
    return j

def compute_dL_dyhat(training_labels, yhat):
    dL_dyhat = (1/(training_labels.shape[0] * 0.5)) * (training_labels/yhat)
    return dL_dyhat

def compute_dyhat_dh(layer):
    w = weights[layer]
    (yhat - training_labels).T * w
    return w

def compute_dz_dw(layer_num, activation_list):
    dz_dw = activation_list[layer_num]
    return dz_dw

def compute_dz_db(layer_num):
    dz_db = biases[layer_num]
    return dz_db

def forward(weights, biases, training_data):
    activation_list = [training_data]
    activation = relu(weights[0], training_data, biases[0])
    activation_list.append(activation)
    i = 1
    while i < len(weights)-1:
        activation = relu(weights[i], activation, biases[i])
        activation_list.append(activation)
        i = i + 1
    yhat = calc_softmax(weights[len(weights)-1], activation, biases[len(biases)-1])
    return yhat, activation_list


def init_weights(input_hidden_layers, unit_count, training_labels):
    weights = []
    cons = (1/unit_count) ** 0.4
    first_w = cons * np.random.randn(training_data.shape[1], unit_count)
    weights.append(first_w)
    last_w = cons * np.random.randn(unit_count, training_labels.shape[1])
    for i in range(input_hidden_layers - 2):
        w = cons * np.random.rand(unit_count, unit_count)
        weights.append(w)
    weights.append(last_w)
    return weights


def init_biases(input_hidden_layers, unit_count, training_labels):
    biases = []
    cons = (1/unit_count) ** 0.4
    last_b = cons * np.zeros(training_labels.shape[1])
    for i in range(input_hidden_layers - 1):
        b = cons * np.zeros(unit_count)
        biases.append(b)
    biases.append(last_b)
    return biases


def backward(weights, biases, input_hidden_layers, batch_size, yhat, activation_list, input_data, labels):
    delta_w = []
    delta_b = []
    g = yhat - labels
    #dL_dw = np.dot(g.T, compute_dz_dw(input_hidden_layers-1, activation_list))
    dL_dw = (1/batch_size) *np.dot(g.T, activation_list[input_hidden_layers-1])
    dL_db = np.mean(g, axis=0)
    delta_w.append(dL_dw)
    delta_b.append(dL_db)
    j = input_hidden_layers-1
    while j >= 1: # Starting from one before last layer
        g = np.dot(g, weights[j].T) * relu_prime(j, activation_list)
        dL_dw = (1/batch_size) * np.dot(g.T, activation_list[j-1])
        dL_db = np.mean(g, axis=0)
        delta_w.append(dL_dw)
        delta_b.append(dL_db)
        j = j - 1
    return delta_w, delta_b


def update_variables(weights, biases, delta_w, delta_b, learning_rate):
    k = 0
    for k in range(len(weights)):
        weights[k] -= learning_rate * delta_w[len(weights)-1-k].T
        biases[k] -= learning_rate * delta_b[len(biases)-1-k]
    return weights, biases


def train(training_data, training_labels, layer_count, unit_count, batch_size, alpha):
    input_hidden_layers = layer_count - 1
    weights = init_weights(input_hidden_layers, unit_count, training_labels)
    #print "weight size", len(weights)   
    biases = init_biases(input_hidden_layers, unit_count, training_labels)
    #print "bias size", len(biases)
    epsilon = 0.01
    data_size = training_data.shape[0]
    random_indexes = np.random.choice(data_size, data_size, replace=False)
    epoch = 0
    while epoch < 80:
        if epoch%20 == 0:
            print "Epoch:", epoch
        i = 0
        while i < data_size:    # Loops every batch size
            data_batch = training_data[random_indexes[i: i+batch_size]]
            label_batch = training_labels[random_indexes[i: i+batch_size]]

            #Performing forward 
            yhat, activation_list = forward(weights, biases, data_batch)

            #Perfoming backward
            delta_w, delta_b = backward(weights, biases, input_hidden_layers, batch_size, yhat, activation_list, data_batch, label_batch)

            # Update weights and biases    
            weights, biases = update_variables(weights, biases, delta_w, delta_b, epsilon)
            i = i + 1
        epoch = epoch + 1

    return weights, biases




"""

def train(training_data, training_labels, layer_count, unit_count, batch_size, alpha):
    input_hidden_layers = layer_count - 1
    weights = init_weights(input_hidden_layers, unit_count, training_labels)
    #print "weight size", len(weights)   
    biases = init_biases(input_hidden_layers, unit_count, training_labels)
    #print "bias size", len(biases)
    epsilon = 0.01
    data_size = training_data.shape[0]
    random_indexes = np.random.choice(data_size, data_size, replace=False)
    epoch = 0
    while epoch < 80:
        if epoch%20 == 0:
            print "Epoch:", epoch
        i = 0
        while i < data_size:    # Loops every batch size
            data_batch = training_data[random_indexes[i: i+batch_size]]
            label_batch = training_labels[random_indexes[i: i+batch_size]]

            #Performing forward 
            yhat, activation_list = forward(weights, biases, data_batch)
            #print "size of activation list", len(activation_list)

            #Perfoming backward
            #g = compute_dL_dyhat(label_batch, yhat)
            delta_w = []
            delta_b = []
            g = yhat - label_batch
            #g = np.dot(compute_dz_dw(layer_index, activation_list).T, g)
            #print "length of A", len(activation_list)
            #print activation_list[0].shape
            #print activation_list[1].shape
            #print activation_list[2].shape
            #print activation_list[3].shape
            #print "len of W", len(weights)
            dL_dw = np.dot(g.T, compute_dz_dw(input_hidden_layers-1, activation_list))
            dL_db = np.mean(g, axis=0)
            delta_w.append(dL_dw)
            delta_b.append(dL_db)
            #print "dimension of g", g.shape
            #print "dw outside of loop", dL_dw.shape
            #print "shape of last activation", activation_list[input_hidden_layers-1].shape
            #print "len of acitvation", len(activation_list)
            #weights[input_hidden_layers-1] -= epsilon * dL_dw.T
            #biases[input_hidden_layers-1] -= epsilon * dL_db
            #print "first weight shape", weights[input_hidden_layers-1].shape
            
            j = input_hidden_layers-1
            while j >= 1: # Starting from one before last layer
                #print "##########################################"
                #print "j", j
                #print "w shape before", weights[j-1].shape
                #print "w shape", weights[j].shape
                #print "w shape after", weights[j+1].shape
                #print "g before", g.shape
                #print "relu prime one after", relu_prime(j+1, activation_list).shape
                #print "relu prime", relu_prime(j, activation_list).shape
                #print "relu prime one before", relu_prime(j-1, activation_list).shape
                g = np.dot(g, weights[j].T) * relu_prime(j, activation_list)
                #print "g after", g.shape
                #print "shape of A[j-1]", activation_list[j-1].shape
                dL_dw = np.dot(g.T, activation_list[j-1])
                dL_db = np.mean(g, axis=0)
                #print "dL_db", dL_db.shape
                delta_w.append(dL_dw)
                delta_b.append(dL_db)
                #weights[j-1] = weights[j-1] - (dL_dw.T * epsilon)
                #biases[j-1] = biases[j-1] - (dL_db * epsilon)
                
                j = j - 1
            
            k = 0
            for k in range(len(weights)):
                weights[k] -= epsilon * delta_w[len(weights)-1-k].T
                biases[k] -= epsilon * delta_b[len(biases)-1-k]
                #print np.linalg.norm(weights[k])
                print weights[k]
                print biases[k]

            i = i + 1

        epoch = epoch + 1
    
    return weights, biases
"""


# Reports the cost for both training data and validation data by running the cost function (loss)
# Inputs: weights w, training data, training labels, validation data, validation labels, alpha,   Output: weight
def reportCosts (w, training_data, training_labels, validation_data, validation_labels, alpha = 0.):
    print "Training cost: {}".format(loss(w, training_data, training_labels, alpha))
    print "Validation cost:  {}".format(loss(w, validation_data, validation_labels, alpha))

# Takes in the weight that we got from training data when compared to training lebels and inputs that eight into softmax (z = wight * testing data)
# Then compares the predicted results with the testing labels and if the highest value of each row (data instance) has the same inidex as the label then
# increments the accuracy counter.
# Inputs: wights w, testing data, testing labels,  Output: accuracy achieved from the model on test data 
def accuracy_function(w, validation_data, validation_labels):
    predictions = calc_softmax(w, testing_data)
    accuracy_counter = 0.0
    for row_index in range(len(predictions)):
        if np.argmax(predictions[row_index]) == np.argmax(testing_labels[row_index]):
            accuracy_counter += 1.0
    accuracy = (accuracy_counter/testing_labels.shape[0]) * 100
    print "accuracy:", accuracy, "%"
    return accuracy





def accuracy(weights, biases, testing_data, testing_labels):
    predictions, activation_lsit = forward(weights, biases, testing_data)
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

    ALPHA = 1E-3
    batch_size = 1000
    layer_count = 5
    unit_count = 30

    weights, biases = train(training_data, training_labels, layer_count, unit_count, batch_size, ALPHA)
    #reportCosts(w, training_data, training_labels, validation_data, validation_labels, ALPHA)
    accuracy(weights, biases, validation_data, validation_labels)




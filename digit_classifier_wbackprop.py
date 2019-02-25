# This code is a two layer neural network that uses the MNIST dataset to train a neural network to classify hand written digits.
# The training data is 55000 (26x28) images of handwritten data in the shape of a flattened 784 dimentional vector.
# Validation data is 5000 images and testing data includes 10000 images.
# The output data of the model includes all the images with probabilities of all digits for each instance. 
# The code uses the cross entropy loss function and stochastic gradient descent to minimize it. The activation function implemented is Softmax
# This code is written in Python 2.7

# Training_data: 55000 x 784
# First weight matrix  = w : 784 x number of hidden layers (30)



import numpy as np

# The softmax function
# Inputs: wights w (784x10), input data (55000x784),  Output: digits: all the input data with probabilities (55000x10)
def calc_softmax(w, activation_matrix, b):
    z_data = np.dot(activation_matrix, w) + b
    yhat = (np.exp(z_data.T) / np.sum(np.exp(z_data), axis=1)).T
    return yhat

def relu(w, training_data, b):
    z_data = np.dot(training_data, w) + b
    activation_matrix = np.maximum(z_data, 0)
    return activation_matrix

# The cross-entropy loss function 
# Inputs: wights w (784x10), input data, labels, alpha: regularization constant,  Output: j: the cost value
def loss(training_labels, yhat):
    j = training_labels * np.log(yhat)
    j = np.mean(j) * (0.5) * (-1)
    #j = j + ((alpha * 0.5) * np.sum(np.dot(w,w.T)))
    return j

# The gradient of the loss functoin
# Inputs: wights w, input data, lebels, alpha: regularization constant,  Output: Jw: gradient of loss function
def loss_gradient(w, training_data, training_labels, alpha):
    activation_matrix = calc_softmax(w, training_data)
    Jw = np.dot(training_data.T, (activation_matrix - training_labels)) 
    Jw = Jw/training_data.shape[0]
    Jw = Jw + (alpha * w)
    return Jw

def compute_dL_dyhat(training_labels, yhat):
    dL_dyhat = (1/(training_labels.shape[0] * 0.5)) * (training_labels/yhat)
    return dL_dyhat

def compute_dyhat_dh(layer):
    w = weights[layer]
    (yhat - training_labels).T * w
    return w

def relu_prime(layer, activation_list):
    a = activation_list[layer]
    return np.where(a < 0, 0, 1)

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



def init_weights(layer_count, unit_count, training_labels):
    weights = []
    first_w = np.random.randn(training_data.shape[1], unit_count) * 0.1
    weights.append(first_w)
    last_w = np.random.randn(unit_count, training_labels.shape[1]) * 0.1
    for i in range(layer_count - 2):
        w = np.random.rand(unit_count, unit_count)
        weights.append(w)
    weights.append(last_w)
    return weights


def init_bias(layer_count, unit_count, training_labels):
    biases = []
    first_b = np.zeros(unit_count)
    biases.append(first_b)
    last_b = np.zeros(training_labels.shape[1])
    for i in range(layer_count - 2):
        b = np.zeros(unit_count)
        biases.append(b)
    biases.append(last_b)
    return biases




def backward(previous, w, layer, activation_list):
    #previous = training_data - training_labels * activation_list[layer]
    dL_dw = previous * w * compute_dh_dz(layer, activation_list) * compute_dz_dw(layer, activation_list)
    dL_db = previous * b * compute_dh_dz(layer, activation_list) * compute_dz_db(layer)
    return dL_dw, dL_db


"""
def train(training_data, training_labels, layer_count, unit_count, batch_size, alpha):
    weights = init_weights(layer_count, unit_count, training_labels)
    biases = init_bias(layer_count, unit_count, training_labels)
    epsilon = 0.1
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
            # Performing backward
            previous_w = np.dot((yhat - label_batch).T, activation_list[layer_count - 2])
            previous_b = np.dot((yhat - label_batch).T, activation_list[layer_count - 2])
            j = layer_count - 3
            while j >= 1: # Starting from one before last layer
                dL_dw, dL_db = backward(previous_w, previous_b, wights[j], j, activation_list)
                previous_w = dL_dw
                previous_b = dL_db
                weights[j] = weights[j] - (dL_dw * alpha)
                biases[j] = biases[j] - (dL_db * alpha)
                j = j - 1
            i = i + batch_size
        epoch = epoch + 1
    return weights, biases
"""

# Approach 2: #########################3

def train(training_data, training_labels, layer_count, unit_count, batch_size, alpha):
    input_hidden_layers = layer_count - 1
    weights = init_weights(input_hidden_layers, unit_count, training_labels)
    #print "weight size", len(weights)   
    biases = init_bias(input_hidden_layers, unit_count, training_labels)
    #print "bias size", len(biases)
    epsilon = 0.1
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
            dL_dw = np.dot(g.T, compute_dz_dw(input_hidden_layers-1, activation_list))
            dL_db = np.mean(g, axis=0)
            delta_w.append(dL_dw)
            delta_b.append(dL_db)
            #weights[input_hidden_layers-1] -= epsilon * dL_dw.T
            #biases[input_hidden_layers-1] -= epsilon * dL_db
            #print "first weight shape", weights[input_hidden_layers-1].shape
            #print "dimension of g", g.shape
            j = input_hidden_layers-1
            while j >= 0: # Starting from one before last layer
                #print "j", j
                #print "shape of relu prime", relu_prime(j-1, activation_list).shape
                #print "w shape", weights[j].shape
                g = np.dot(g, weights[j].T) * relu_prime(j-1, activation_list)
                #print "g shape", g.shape
                dL_dw = np.dot(g.T, activation_list[j-1])
                dL_db = np.mean(g, axis=0)
                delta_w.append(dL_dw)
                delta_b.append(dL_db)
                #weights[j-1] = weights[j-1] - (dL_dw.T * epsilon)
                #biases[j-1] = biases[j-1] - (dL_db * epsilon)
                print np.linalg.norm(weights[j-1])
                j = j - 1
            
            k = 0
            for k in range(len(weights)):
                weights[k] -= epsilon * delta_w[len(weights)-1-k]
                biases[k] -= epsilon * delta_w[len(biases)-1-k]

            i = i + 1

        epoch = epoch + 1
    
    return weights, biases

        



            



# Gradient Descent: For every epoch it divides the data into smaller batches and runs gradient descient on it (updates the weights with epsilon(learning rate))
# The function generates an array of random numbers between 0 and data_size and for every batch it takes the random numbers and uses the corresponding 
# values in training data to randomize the data
# Inputs: input data, labels, batch size, alpha: regularization constant,  Output: w: the final updated weight (the weight when the model converges)
def gradientDescent(training_data, training_labels, batch_size, layer_count, unit_count, alpha = 0.):
    w = init_weights(layer_count, unit_count, training_labels)
    b = init_bias(layer_count, unit_count, training_labels)
    epsilon = 0.001   # Learning Rate
    data_size = training_data.shape[0]
    random_indexes = np.random.choice(data_size, data_size, replace=False)
    epoch = 0
    while epoch < 200:
        if epoch%20 == 0:
            print "Epoch:", epoch
        i = 0
        while i < data_size:    # Loops every batch size
            data_batch = training_data[random_indexes[i: i+batch_size]]
            label_batch = training_labels[random_indexes[i: i+batch_size]]
            gradient = loss_gradient(w, data_batch, label_batch, alpha)
            w = w - (gradient * epsilon)
            i = i + batch_size
        epoch = epoch + 1
    return w



# A wrapper for gradient discent
# Inputs: input data, labels, batch size, aplha,  Output: w: the final weight when the model converges
#def wrapper(training_data, training_labels, layer_count, unit_count, alpha):
    #return train(training_data, training_labels, layer_count, unit_count, alpha)

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




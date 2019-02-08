# This code is a two layer neural network that uses the MNIST dataset to train a neural network to classify hand written digits.
# The training data is 55000 (26x28) images of handwritten data in the shape of a flattened 784 dimentional vector.
# Validation data is 5000 images and testing data includes 10000 images.
# The output data of the model includes all the images with probabilities of all digits for each instance. 
# The code uses the cross entropy loss function and stochastic gradient descent to minimize it. The activation function implemented is Softmax


import numpy as np

# The softmax function
# Inputs: wights w (784x10), input data (55000x784),  Output: digits: all the input data with probabilities (55000x10)
def calc_softmax(w, training_data):
    z_data = np.dot(training_data, w)
    digits = (np.exp(z_data.T) / np.sum(np.exp(z_data), axis=1)).T
    return digits

# The cross-entropy loss function 
# Inputs: wights w (784x10), input data, labels, alpha: regularization constant,  Output: j: the cost value
def loss(w, training_data, training_labels, alpha):
    digits = calc_softmax(w, training_data)
    j = training_labels * np.log(digits)
    j = np.mean(j) * (0.5) * (-1)
    j = j + ((alpha * 0.5) * np.sum(np.dot(w,w.T)))
    return j

# The gradient of the loss functoin
# Inputs: wights w \, input data, lebels, alpha: regularization constant,  Output: Jw: gradient of loss function
def loss_gradient(w, training_data, training_labels, alpha):
    digits = calc_softmax(w, training_data)
    Jw = np.dot(training_data.T, (digits - training_labels)) 
    Jw = Jw/training_data.shape[0]
    Jw = Jw + (alpha * w)
    return Jw

# Gradient Descent: For every epoch it divides the data into smaller batches and runs gradient descient on it (updates the weights with epsilon(learning rate))
# The function generates an array of random numbers between 0 and data_size and for every batch it takes the random numbers and uses the corresponding 
# values in training data to randomize the data
# Inputs: input data, labels, batch size, alpha: regularization constant,  Output: w: the final updated weight (the weight when the model converges)
def gradientDescent(training_data, training_labels, batch_size, alpha = 0.):
    w = np.zeros((training_data.shape[1], 10))  # Or set to random vector
    epsilon = 0.1   # Learning Rate
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
def wrapper(training_data, training_labels, batch_size, alpha):
    return gradientDescent(training_data, training_labels, batch_size, alpha)

# Reports the cost for both training data and validation data by running the cost function (loss)
# Inputs: weights w, training data, training labels, validation data, validation labels, alpha,   Output: weight
def reportCosts (w, training_data, training_labels, validation_data, validation_labels, alpha = 0.):
    print "Training cost: {}".format(loss(w, training_data, training_labels, alpha))
    print "Validation cost:  {}".format(loss(w, validation_data, validation_labels, alpha))

# Takes in the wight that we got from training data when compared to training lebels and inputs that eight into softmax (z = wight * testing data)
# Then compares the predicted results with the testing labels and if the highest value of each row (data instance) has the same inidex as the label then
# increments the accuracy counter.
# Inputs: wights w, testing data, testing labels,  Output: accuracy achieved from the model on test data 
def accuracy_function(w, testing_data, testing_labels):
    predictions = calc_softmax(w, testing_data)
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

    w = wrapper(training_data, training_labels, batch_size, ALPHA)
    reportCosts(w, training_data, training_labels, validation_data, validation_labels, ALPHA)
    accuracy_function(w, testing_data, testing_labels)




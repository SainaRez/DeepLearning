import keras 
import numpy as np
from keras.datasets import mnist


# load mnist data, deivid both training and test by apriximately 3 and normalize them
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train[40000:].astype('float32') / 255
    y_train = y_train[40000:]

    x_test = x_test[7000:].astype('float32') / 255
    y_test = y_test[7000:]
    

    return x_train, y_train, x_test, y_test



# A functional dense network
def dense_network(x_train, y_train, x_test, y_test):
    
    label_number = len(np.unique(y_train))
    input_ = keras.layers.Input(shape=[28, 28])                         #(?, 28, 28)
    flatten = keras.layers.Flatten(input_shape=[28, 28])(input_)        #(?, 784)

    # Add the dense layers
    hidden1 = keras.layers.Dense(16384, activation="relu")(flatten)     #(?, 16384)
    hidden2 = keras.layers.Dense(512, activation='relu')(hidden1)       #(?, 512)
    hidden3 = keras.layers.Dense(784, activation='relu')(hidden2)       #(?, 784)

    reshap = keras.layers.Reshape((28, 28))(hidden3)                    #(?, 28, 28)
    concat_ = keras.layers.Concatenate()([input_, reshap])              #(?, 28, 28) + (?, 28, 28) = (?, 28, 56)

    flatten2 = keras.layers.Flatten(input_shape=[28, 28])(concat_)                #(?, 1568)
    output = keras.layers.Dense(label_number, activation='softmax')(flatten2)     #(?, 10)
    model = keras.Model(inputs=[input_], outputs=[output] )

    # Compile and fit the model
    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    h = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


    # Get the accuracy of the model on test data
    score = model.evaluate(x_test, y_test, verbose=0)
    print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))




if __name__=='__main__':
    
    # load data
    x_train, y_train, x_test, y_test = load_data()

    dense_network(x_train, y_train, x_test, y_test)
    

# CNN network
#x_train, y_train), (x_test, y_test) = mnist.load_data()
#num_labels = len(np.unique(y_train))
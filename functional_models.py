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


# Get the accuracy of the model on test data
def get_accuracy(x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))


# Adding the layers
def create_dense_model(class_num, activation, final_activation, image_shape):

    input_ = keras.layers.Input(shape=image_shape)                         #(?, 28, 28)
    flatten = keras.layers.Flatten(input_shape=[28, 28])(input_)        #(?, 784)

    # Add the dense layers
    hidden1 = keras.layers.Dense(16384, activation = activation)(flatten)     #(?, 16384)
    hidden2 = keras.layers.Dense(512, activation = activation)(hidden1)       #(?, 512)
    hidden3 = keras.layers.Dense(784, activation = activation)(hidden2)       #(?, 784)

    reshap = keras.layers.Reshape(image_shape)(hidden3)                    #(?, 28, 28)
    concat_ = keras.layers.Concatenate()([input_, reshap])              #(?, 28, 28) + (?, 28, 28) = (?, 28, 56)

    flatten2 = keras.layers.Flatten(input_shape=image_shape)(concat_)                #(?, 1568)
    output = keras.layers.Dense(class_num, activation = final_activation)(flatten2)     #(?, 10)
    model = keras.Model(inputs=[input_], outputs=[output] )

    return model


# A functional dense network
def dense_network(x_train, y_train, x_test, y_test):
    
    # parameters
    label_number = len(np.unique(y_train))
    activation = 'relu'
    final_activation = 'softmax'
    image_shape = [28, 28]
    
    # create model
    model = create_dense_model(label_number, activation, final_activation, image_shape)
    
    # Compile and fit the model
    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    h = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    get_accuracy(x_test, y_test)


# Additing the layers
def create_cnn_model(input_shape, kernel_size, filters, dropout):

    input_ = keras.Input(shape=input_shape)
    conv1 = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(input_)
    pooling_1 = keras.layers.MaxPooling2D()(conv1)
    conv2 = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(pooling_1)
    pooling_2 = keras.layers.MaxPooling2D()(conv2)
    conv3 = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(pooling_2)
    
    # convert image to vector before connecting to dense layer
    flatten = keras.layers.Flatten()(conv3)
    
    # dropout regularization
    dropout = keras.layers.Dropout(dropout)(flatten)
    outputs = keras.layers.Dense(10, activation='softmax')(dropout)
    
    # build the model
    model = keras.Model(inputs=input_, outputs=outputs)
    model.summary()
    return model


def cnn_network(x_train, y_train, x_test, y_test):
    
    # reshape the data
    x_train = np.reshape(x_train,[-1, 28, 28, 1])
    x_test = np.reshape(x_test,[-1, 28, 28, 1])
    print(x_train.shape)

    # Parameters
    input_shape = (28, 28, 1)
    kernel_size = 3
    filters = 64
    dropout = 0.3

    model = create_cnn_model(input_shape, kernel_size, filters, dropout)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

    get_accuracy(x_test, y_test)




if __name__=='__main__':
    
    # load data
    x_train, y_train, x_test, y_test = load_data()

    #dense_network(x_train, y_train, x_test, y_test)
    cnn_network(x_train, y_train, x_test, y_test)

# CNN network
#x_train, y_train), (x_test, y_test) = mnist.load_data()
#num_labels = len(np.unique(y_train))
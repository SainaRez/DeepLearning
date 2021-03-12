import keras 
import numpy as np
from keras.datasets import mnist


class Data:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_data():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        self.x_train = x_train[40000:].astype('float32') / 255
        self.y_train = y_train[40000:]

        self.x_test = x_test[7000:].astype('float32') / 255
        self.y_test = y_test[7000:]



class Train():
    def __init__(self):
    self.cnn

    def call_cnn():
        pass

class FunctionalNetworks:
    def __init__(self, model):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = model

    
    # Get the accuracy of the model on test data
    def get_accuracy(x_test, y_test, model):
        score = model.evaluate(x_test, y_test, verbose=0)
        print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))
        


class Dense(FunctionalNetworks):
    def __init__(self):
        super(self).__init__()
        self.model = None
    
    def create_model():

        # parameters
        label_number = len(np.unique(y_train))
        activation = 'relu'
        final_activation = 'softmax'
        image_shape = [28, 28]

        input_ = keras.layers.Input(shape=image_shape)                         #(?, 28, 28)
        flatten = keras.layers.Flatten(input_shape=[28, 28])(input_)        #(?, 784)

        # Add the dense layers
        hidden1 = keras.layers.Dense(1024, activation = activation)(flatten)     #(?, 1024)
        hidden2 = keras.layers.Dense(512, activation = activation)(hidden1)       #(?, 512)
        hidden3 = keras.layers.Dense(784, activation = activation)(hidden2)       #(?, 784)

        output = keras.layers.Dense(label_number, activation = final_activation)(hidden3)     #(?, 10)
        self.model = keras.Model(inputs=[input_], outputs=[output])
        self.model.summary()

        # Compile and fit the model
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        self.model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

        


class CNN(FunctionalNetworks):
    def __init__(self):
        super(self).__init__()
        self.model = None

    def create_model():

        y_train = keras.utils.to_categorical(y_train)
        y_test = keras.utils.to_categorical(y_test)
    
    
        # reshape the data
        x_train = np.reshape(x_train,[-1, 28, 28, 1])
        x_test = np.reshape(x_test,[-1, 28, 28, 1])

        print(x_train.shape)
        print(x_train.shape)

         # Parameters
        input_shape = (28, 28, 1)
        kernel_size = 3
        filters = 64
        dropout = 0.3


        input_ = keras.layers.Input(shape=input_shape)
        conv1 = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(input_)
        pooling_1 = keras.layers.MaxPooling2D()(conv1)
        conv2 = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(pooling_1)
        pooling_2 = keras.layers.MaxPooling2D()(conv2)
        conv3 = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(pooling_2)
    
        # convert image to vector before conne`cting to dense layer
        flatten = keras.layers.Flatten()(conv3)
    
        # dropout regularization
        dropout = keras.layers.Dropout(dropout)(flatten)
        output_ = keras.layers.Dense(10, activation='softmax')(dropout)


        # build the model
        self.model = keras.Model(inputs=input_, outputs=output_)
        self.model.summary()

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, validation_data=[x_test, y_test], epochs=5)

        


class RNN(FunctionalNetworks):
    def __init__(self):
        super(self).__init__()
        self.model = None

    def create_mode():
        input_shape = [28, 28]

        input_ = keras.layers.Input(shape=input_shape)
        lstm = keras.layers.LSTM(128)(input_)  #(timesteps, number of features in each timestep)
        output_ = keras.layers.Dense(10, activation='softmax')(lstm)

        self.model = keras.Model(inputs=input_, outputs=output_)
        self.model.summary()

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)




# load mnist data, deivid both training and test by apriximately 3 and normalize them
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train[40000:].astype('float32') / 255
    y_train = y_train[40000:]

    x_test = x_test[7000:].astype('float32') / 255
    y_test = y_test[7000:]
    

    return x_train, y_train, x_test, y_test


# Get the accuracy of the model on test data
def get_accuracy(x_test, y_test, model):
    score = model.evaluate(x_test, y_test, verbose=0)
    print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))


# Adding the layers
def create_dense_model(class_num, activation, final_activation, image_shape):

    input_ = keras.layers.Input(shape=image_shape)                         #(?, 28, 28)
    flatten = keras.layers.Flatten(input_shape=[28, 28])(input_)        #(?, 784)

    # Add the dense layers
    hidden1 = keras.layers.Dense(1024, activation = activation)(flatten)     #(?, 1024)
    hidden2 = keras.layers.Dense(512, activation = activation)(hidden1)       #(?, 512)
    hidden3 = keras.layers.Dense(784, activation = activation)(hidden2)       #(?, 784)

    # reshape = keras.layers.Reshape(image_shape)(hidden3)                    #(?, 28, 28)
    # concat_ = keras.layers.Concatenate()([input_, reshape])              #(?, 28, 28) + (?, 28, 28) = (?, 28, 56)

    # flatten2 = keras.layers.Flatten(input_shape=image_shape)(reshape)                #(?, 1568)
    output = keras.layers.Dense(class_num, activation = final_activation)(hidden3)     #(?, 10)
    model = keras.Model(inputs=[input_], outputs=[output])
    model.summary()

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
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    get_accuracy(x_test, y_test, model)

 
# Adding the layers
def create_cnn_model(input_shape, kernel_size, filters, dropout):

    input_ = keras.layers.Input(shape=input_shape)
    conv1 = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(input_)
    pooling_1 = keras.layers.MaxPooling2D()(conv1)
    conv2 = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(pooling_1)
    pooling_2 = keras.layers.MaxPooling2D()(conv2)
    conv3 = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(pooling_2)
    
    # convert image to vector before conne`cting to dense layer
    flatten = keras.layers.Flatten()(conv3)
    
    # dropout regularization
    dropout = keras.layers.Dropout(dropout)(flatten)
    output_ = keras.layers.Dense(10, activation='softmax')(dropout)


    # build the model
    model = keras.Model(inputs=input_, outputs=output_)
    model.summary()
    return model


def cnn_network(x_train, y_train, x_test, y_test):
    
    
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    
    
    # reshape the data
    x_train = np.reshape(x_train,[-1, 28, 28, 1])
    x_test = np.reshape(x_test,[-1, 28, 28, 1])

    print(x_train.shape)
    print(x_train.shape)

    # Parameters
    input_shape = (28, 28, 1)
    kernel_size = 3
    filters = 64
    dropout = 0.3

    model = create_cnn_model(input_shape, kernel_size, filters, dropout)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=[x_test, y_test], epochs=5)

    get_accuracy(x_test, y_test, model)


def create_rnn_model(input_shape):

    input_ = keras.layers.Input(shape=input_shape)
    lstm = keras.layers.LSTM(128)(input_)
    output_ = keras.layers.Dense(10, activation='softmax')(lstm)

    model = keras.Model(inputs=input_, outputs=output_)
    model.summary()
    return model


def rnn_network(x_train, y_train,  x_test, y_test):
    input_shape = [28, 28]
    model = create_rnn_model(input_shape)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

    get_accuracy(x_test, y_test, model)



if __name__=='__main__':
    
    # load data
    #x_train, y_train, x_test, y_test = load_data()

    #dense_network(x_train, y_train, x_test, y_test)
    #cnn_network(x_train, y_train, x_test, y_test)
    #rnn_network(x_train, y_train,  x_test, y_test)
    d = Dense()
    d.load_data()


# CNN network
#x_train, y_train), (x_test, y_test) = mnist.load_data()
#num_labels = len(np.unique(y_train))
import numpy as npy
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt



def pre_process(train_images, train_lables, test_images, test_labels):

    train_images = train_images/255.0
    test_images = test_images/255.0

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()
    return train_images, train_labels, test_images, test_labels

def build_model(train_images, train_labels, test_images, test_labels):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorial_crossentropy', matrics=['accuracy'])
    return model





if __name__ == "__main__":

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    print train_images.shape
    print train_labels.shape
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    train_images, train_labels, test_images, test_labels = pre_process(train_images, train_labels, test_images, test_labels)

    model = build_model(train_images, train_labels, test_images, test_labels)
    model.fit(train_images, train_labels, epochs=20)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("TEST LOSS:", test_loss, "TEST ACCURACY:", test_acc)

    """
    predictions = model.predict(test_images)
    for image in range(len(predictions)):
        np.argmax(image)
    """
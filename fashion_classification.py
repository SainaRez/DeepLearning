import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def pre_process(train_images, train_labels, test_images, test_labels):

    train_images = train_images/255.0
    test_images = test_images/255.0

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

    """
    validation_images = train_images[50000:60001]
    train_images = train_images[0:50001]
    validation_labels = train_labels[50000:60001]
    train_labels = train_labels[0:50001]
    """

    return train_images, train_labels, test_images, test_labels


def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def CNN_model(batch_size):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=2,
                                  padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(filters=64, kernel_size=2,
                                  padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def calc_model_accuracy(train_images, train_labels, test_images, test_labels, batch_size):
    print "Epochs:", "10", "Batch size:", batch_size
    model = build_model()
    #model.fit(train_images, train_labels, epochs=10)
    model.fit(train_images, train_labels, batch_size, epochs=10,
              validation_split=0.1)
    test_loss, test_acc = model.evaluate(
        x=test_images, y=test_labels, batch_size=batch_size)
    print "TEST ACCURACY:", test_acc*100.0
    print "##################################"

    predictions = model.predict(test_images)

    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_labels)
    plt.show()


def calc_cnn_accuracy(train_images, train_labels, test_images, test_labels, batch_size):
    print "Epochs:", "10", "Batch size:", batch_size

    original = test_images
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    cnn_model = CNN_model(batch_size)
    cnn_model.fit(train_images, train_labels, batch_size, epochs=10,
                  validation_split=0.1)

    score = cnn_model.evaluate(test_images, test_labels, verbose=0, batch_size=batch_size)
    print 'Test accuracy:', score[1]*100
    print "##################################"

    predictions = cnn_model.predict(test_images)

    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, original)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_labels)
    plt.show()


if __name__ == "__main__":

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    """
  plt.figure()
  plt.imshow(train_images[0])
  plt.colorbar()
  plt.grid(False)
  plt.show()
  """

    train_images, train_labels, test_images, test_labels = pre_process(
        train_images, train_labels, test_images, test_labels)

    print train_images.shape
    print train_labels.shape
    batch_size = 100

    calc_model_accuracy(train_images, train_labels,
                      test_images, test_labels, batch_size)
    calc_cnn_accuracy(train_images, train_labels,
                      test_images, test_labels, batch_size)

    """
  model = build_model()
  print "Epochs:", "10", "Batch size:", "1000"
  model.fit(train_images, train_labels, epochs=10)
  test_loss, test_acc = model.evaluate(x=test_images, y=test_labels, batch_size=1000)
  print "TEST ACCURACY:", test_acc*100.0

  predictions = model.predict(test_images)

  num_rows = 5
  num_cols = 3
  num_images = num_rows*num_cols
  plt.figure(figsize=(2*2*num_cols, 2*num_rows))
  for i in range(num_images):
      plt.subplot(num_rows, 2*num_cols, 2*i+1)
      plot_image(i, predictions, test_labels, test_images)
      plt.subplot(num_rows, 2*num_cols, 2*i+2)
      plot_value_array(i, predictions, test_labels)
  plt.show()

  cnn_model = CNN_model()
  cnn_model.fit(train_images, train_labels, batch_size=64, epochs=10, validation_data=(x_valid, y_valid),
        callbacks=[checkpointer])
  
  cnn_model = CNN_model()
  score = cnn_model.evaluate(test_images, test_labels, verbose=0)
  print 'Test accuracy:', score[1]
  """

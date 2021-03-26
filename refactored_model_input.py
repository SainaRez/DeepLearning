import os
import numpy as np
import h5py

from tensorflow_datasets.core import lazy_imports_lib
import tensorflow_datasets.public_api as tfds

import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display



class Data:

    def __init__(self, data_path, batch):
        self.path = data_path
        self.batch_size = batch
        
        #self.files_ds = None
        self.waveform_dataset = None
        #self.spectrogram_dataset = None

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        
        self.load_data = self.LoadData()
        self.preprocess_data = self.PreprocessData()

        


    # Set seed for experiment reproducibility
    def random_seed(self):
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
    def load_data_wrapper(self):
        self.load_data.get_path_download(self.path)
        self.load_data.get_classes()
        self.load_data.divide_data()
        self.load_data.get_audio_label_pairs()
        self.waveform_dataset = self.load_data.waveform_ds
        self.load_data.vis_audio_sample()
        self.validation_dataset = self.load_data.val_files
        self.test_dataset = self.load_data.test_files


    def preprocess_data_wrapper(self):
        self.preprocess_data.wave_dataset = self.waveform_dataset
        self.preprocess_data.classes = self.load_data.classes
        #Plotting some data about spectrogram
        self.preprocess_data.vis_wave_spectrogram()
        
        self.preprocess_data.create_spect_dataset()
        self.preprocess_data.vis_spect_sample()
        self.train_dataset = self.preprocess_data.spectrogram_ds

    def preprocess_dataset(self, files):
            files_ds = tf.data.Dataset.from_tensor_slices(files)
            output_ds = files_ds.map(self.load_data.get_waveform_and_label, num_parallel_calls=self.preprocess_data.AUTOTUNE)
            output_ds = output_ds.map(self.preprocess_data.get_spectrogram_and_label_id,  num_parallel_calls=self.preprocess_data.AUTOTUNE)
            return output_ds

    def preprocess_train(self):
        self.train_dataset = self.train_dataset.batch(self.batch_size)
        self.train_dataset = self.train_dataset.cache().prefetch(self.preprocess_data.AUTOTUNE)
    
    def preprocess_validation(self):
        self.validation_dataset = self.preprocess_dataset(self.validation_dataset)
        self.validation_dataset = self.validation_dataset.batch(self.batch_size)
        self.validation_dataset = self.validation_dataset.cache().prefetch(self.preprocess_data.AUTOTUNE)

    def preprocess_test(self):
        self.test_dataset = self.preprocess_dataset(self.test_dataset)



                


    # Class to load the data and return the audio-label dataset
    
    class LoadData:
        def __init__(self):
            self.data_dir = None
            self.classes = None
            self.all_files = None

            self.train_files = None
            self.val_files = None
            self.test_files = None

            self.files_ds = None
            self.waveform_ds = None
            
            

        # This function gets the data from the given path. If the path doesn't exist, it downloads it from the given url
        def get_path_download(self, input_path):
            self.data_dir = pathlib.Path(input_path)
            if not self.data_dir.exists():
                tf.keras.utils.get_file(
                'mini_speech_commands.zip',
                origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                extract=True,
                cache_dir='.', cache_subdir='data')


        def get_classes(self):
            commands = np.array(tf.io.gfile.listdir(str(self.data_dir)))
            commands = commands[commands != 'README.md']
            print('Commands:', commands)
            self.classes = commands

            # printing some stats about the data
            self.all_files = tf.io.gfile.glob(str(self.data_dir) + '/*/*')
            self.all_files = tf.random.shuffle(self.all_files)
            num_samples = len(self.all_files)
            print('Number of total examples:', num_samples)
            print('Number of examples per label:',
                len(tf.io.gfile.listdir(str(self.data_dir/self.classes[0]))))
            print('Example file tensor:', self.all_files[0])

        # Dividing the data into train, eval and test
        def divide_data(self):
            
            self.train_files = self.all_files[:6400]
            self.val_files = self.all_files[6400: 6400 + 800]
            self.test_files = self.all_files[-800:]

            print('Training set size', len(self.train_files))
            print('Validation set size', len(self.val_files))
            print('Test set size', len(self.test_files))
            return

        
        

        #  take in the filename of the WAV file and output a tuple containing the audio and labels for supervised training
        def get_waveform_and_label(self, file_path):

            def _decode_audio(audio_binary):
                audio, _ = tf.audio.decode_wav(audio_binary)
                return tf.squeeze(audio, axis=-1)

        
            # Get the label from WAV file
            def _get_label(file_path):
                parts = tf.strings.split(file_path, os.path.sep)
                # Note: You'll use indexing here instead of tuple unpacking to enable this
                # to work in a TensorFlow graph.
                return parts[-2]


            label = _get_label(file_path)
            audio_binary = tf.io.read_file(file_path)
            waveform = _decode_audio(audio_binary)
            print("shape of the waveform: ", waveform.shape)
            return waveform, label

        
        def get_audio_label_pairs(self):
            # apply process_path to build training set to extract the audio-label pairs and check the results
            AUTOTUNE = tf.data.AUTOTUNE
            self.files_ds = tf.data.Dataset.from_tensor_slices(self.train_files)
            print("the datatset: ", self.files_ds)
            self.waveform_ds = self.files_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)

        def vis_audio_sample(self):
            
            # Examine a few of the data with their labels
            rows = 3
            cols = 3
            n = rows*cols
            fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
            for i, (audio, label) in enumerate(self.waveform_ds.take(n)):
                r = i // cols
                c = i % cols
                ax = axes[r][c]
                ax.plot(audio.numpy())
                ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
                label = label.numpy().decode('utf-8')
                ax.set_title(label)
            
            return

         

    # Class to preprocess the data
    class PreprocessData:
        
        def __init__(self):
            self.classes = None
            self.AUTOTUNE = tf.data.AUTOTUNE

            self.spectrogram_ds = None
            self.wave_dataset = None

        
        def add_padding_to_audio(self, waveform):
            # Padding for files with less than 16000 samples
            zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

            # Concatenate audio with padding so that all audio clips will be of the
            # same length
            waveform = tf.cast(waveform, tf.float32)

            # adding the padding at the end of the data
            equal_length_waves = tf.concat([waveform, zero_padding], 0)
            return equal_length_waves


        # def create_equal_waveform_dataset(self):
        #     AUTOTUNE = tf.data.AUTOTUNE
        #     self.equal_waveform_ds = waveform_ds.map(self.get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
        
              
        # Converting the files to spectrogram, data shape = (124, 129)
        def get_spectrogram(self, waveform):

            equal_length_waves = self.add_padding_to_audio(waveform)

            # stft splits the signal into windows of time and runs a Fourier transform on each window
            spectrogram = tf.signal.stft(equal_length_waves, frame_length=255, frame_step=128)

            # stft returns magnitude and phase and we only use magnitude (tf.abs)
            spectrogram = tf.abs(spectrogram)
            return spectrogram

        def spectrogram_stats(self):
            for waveform, label in self.wave_dataset.take(10):
                label = label.numpy().decode('utf-8')
                spectrogram = self.get_spectrogram(waveform)

            print('Label:', label)
            print('Waveform shape:', waveform.shape)
            print('Spectrogram shape:', spectrogram.shape)
            print('Audio playback')
            display.display(display.Audio(waveform, rate=16000))
            return spectrogram, waveform

            
        # Plot the spectogram in the log scale and plot the wav file
        def plot_spectrogram(self, spectrogram, ax):
            # Convert to frequencies to log scale and transpose so that the time is
            # represented in the x-axis (columns).
            log_spec = np.log(spectrogram.T)
            # log_spec.shape = (129, 124)
            height = log_spec.shape[0]    # 129
            X = np.arange(16000, step=height + 1)
            Y = range(height)
            ax.pcolormesh(X, Y, log_spec, shading='auto')



        def vis_wave_spectrogram(self):
            spectrogram, waveform = self.spectrogram_stats()
            fig, axes = plt.subplots(2, figsize=(12, 8))
            timescale = np.arange(waveform.shape[0])
            axes[0].plot(timescale, waveform.numpy())
            axes[0].set_title('Waveform')
            axes[0].set_xlim([0, 16000])
            self.plot_spectrogram(spectrogram.numpy(), axes[1])
            axes[1].set_title('Spectrogram')
            plt.show()


        # Transform the waveform dataset to have spectrogram images and their corresponding labels as integer IDs
        def get_spectrogram_and_label_id(self, audio, label):
            spectrogram = self.get_spectrogram(audio)

            # Adding a 1 at the end of the array (axis -1)
            spectrogram = tf.expand_dims(spectrogram, -1)
            # Not sure what is going on here??
            label_id = tf.argmax(label == self.classes)
            return spectrogram, label_id


        def create_spect_dataset(self):
            self.spectrogram_ds = self.wave_dataset.map(self.get_spectrogram_and_label_id, num_parallel_calls=self.AUTOTUNE)

            
        def vis_spect_sample(self):   
            # Examine the spectrogram images from the spectrogram_ds for different samples (9 in this case)
            rows = 3
            cols = 3
            n = rows*cols
            fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
            for i, (spectrogram, label_id) in enumerate(self.spectrogram_ds.take(n)):
                r = i // cols
                c = i % cols
                ax = axes[r][c]
                self.plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
                ax.set_title(self.classes[label_id.numpy()])
                ax.axis('off')

            plt.show()


class Network:

    def __init__(self):
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.classes = None

        
        self.optimizer = None
        self.loss = None
        self.EPOCHS = None
        self.input_shape = None
        
        self.history = None

        self.cnn = self.CNN(self)
        self.rnn = self.RNN()

    def set_data(self, train, val, test, classes):
        self.train_data = train
        self.val_data = val
        self.test_data = test
        self.classes = classes
        for sound, label in self.train_data.take(1):
            self.input_shape = sound[0].shape
        print('Input shape:', self.input_shape)

    def run_model(self):
        self.cnn.create_model()
        self.cnn.fit_model()
    
    
    def predict(self):
        test_audio = []
        test_labels = []

        for audio, label in self.test_data:
            test_audio.append(audio.numpy())
            test_labels.append(label.numpy())

        test_audio = np.array(test_audio)
        test_labels = np.array(test_labels)

        pred = self.cnn.model.predict(test_audio)
        y_pred = np.argmax(pred, axis=1)
        y_true = test_labels


        test_acc = sum(y_pred == y_true) / len(y_true)
        print(f'Test set accuracy: {test_acc:.0%}')
        return y_pred, y_true

    def confusion_matrix(self, y_pred, y_true):
        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx, xticklabels=self.classes, yticklabels=self.classes,
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()

    
    class CNN:
        def __init__(self, network_instance):
            self.model = None
            self.network_instance = network_instance


        def create_model(self):
            # The adapt method, when called on the training data, calculates mean and standard deviation
            norm_layer = preprocessing.Normalization()
            norm_layer.adapt(self.network_instance.train_data.map(lambda x, _: x))

            # Add all the layers
            input_ = layers.Input(shape=self.network_instance.input_shape)  # (124, 129, 1)
            resize = preprocessing.Resizing(32, 32)(input_)
            norm_layer = norm_layer(resize)
            conv1 = layers.Conv2D(32, 3, activation='relu')(norm_layer)
            pooling_1 = layers.MaxPooling2D()(conv1)
            conv2 = layers.Conv2D(64, 3, activation='relu')(pooling_1)
            pooling_2 = layers.MaxPooling2D()(conv2)
            dropout_1 = layers.Dropout(0.25)(pooling_2)
            flatten = layers.Flatten()(dropout_1)
            dense_1 = layers.Dense(128, activation='relu')(flatten)
            dropout_2 = layers.Dropout(0.5)(dense_1)
            output_ = layers.Dense(len(self.network_instance.classes))(dropout_2)


            # build the model
            self.model = tf.keras.Model(inputs=input_, outputs=output_)
            self.model.summary()


        def fit_model(self):
            # Compile and fit the model
            self.model.compile(optimizer=tf.keras.optimizers.Adam(), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

            self.history = self.model.fit(self.network_instance.train_data, validation_data=self.network_instance.val_data, epochs=self.network_instance.EPOCHS,
                                callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2))

            return

        def vis_model_history(self):
            metrics = self.history.history
            plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
            plt.legend(['loss', 'val_loss'])
            plt.show()


    class RNN:
        def __init__(self):
            pass




if __name__=="__main__":
    
    path = 'data/mini_speech_commands'
    data = Data(path, 64)
    data.random_seed()
    data.load_data_wrapper()
    data.preprocess_data_wrapper()
    data.preprocess_train()
    data.preprocess_validation()
    data.preprocess_test()

    network = Network()
    network.optimizer = tf.keras.optimizers.Adam()
    network.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    network.EPOCHS = 10
    network.set_data(data.train_dataset, data.validation_dataset, data.test_dataset, data.load_data.classes)
    network.run_model()
    predicted_y, true_y = network.predict()
    network.confusion_matrix(predicted_y, true_y)

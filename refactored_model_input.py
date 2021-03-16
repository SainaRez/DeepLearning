import os
import numpy as np

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

    def __init__(self, data_path):
        self.path = data_path
        
        #self.files_ds = None
        self.waveform_dataset = None
        
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
            self.spectrogram = None
            self.equal_length = None

        # def add_padding_to_audio(waveform):
        #     # Padding for files with less than 16000 samples
        #     zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

        #     # Concatenate audio with padding so that all audio clips will be of the
        #     # same length
        #     waveform = tf.cast(waveform, tf.float32)

        #     # adding the padding at the end of the data
        #     self.equal_length = tf.concat([waveform, zero_padding], 0)
        #     return


    #     # Converting the files to spectrogram, data shape = (124, 129)
    #     def get_spectrogram(waveform):
    #         # Padding for files with less than 16000 samples
    #         zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    #         # Concatenate audio with padding so that all audio clips will be of the
    #         # same length
    #         waveform = tf.cast(waveform, tf.float32)

    #         # adding the padding at the end of the data
    #         equal_length = tf.concat([waveform, zero_padding], 0)

    #         # stft splits the signal into windows of time and runs a Fourier transform on each window
    #         self.spectrogram = tf.signal.stft(
    #             equal_length, frame_length=255, frame_step=128)

    #         # stft returns magnitude and phase and we only use magnitude (tf.abs)
    #         self.spectrogram = tf.abs(self.spectrogram)

    #         return spectrogram

    # def spectrogram_wrapper():
    #     for waveform, label in waveform_ds.take(10):
    #         label = label.numpy().decode('utf-8')
    #         spectrogram = get_spectrogram(waveform)

    #     print('Label:', label)
    #     print('Waveform shape:', waveform.shape)
    #     print('Spectrogram shape:', spectrogram.shape)
    #     print('Audio playback')
    #     display.display(display.Audio(waveform, rate=16000))

        
        
    #     # Plot the spectogram in the log scale and plot the wav file
    #     def plot_spectrogram(ax):
    #         # Convert to frequencies to log scale and transpose so that the time is
    #         # represented in the x-axis (columns).
    #         log_spec = np.log(self.spectrogram.T)
    #         #print("This is log spec: ", log_spec)
    #         # print("##############", log_spec.shape)   #(129, 124)
    #         height = log_spec.shape[0]    # 129
    #         X = np.arange(16000, step=height + 1)
    #         Y = range(height)
    #         ax.pcolormesh(X, Y, log_spec)

    # def create_plot():
    #     fig, axes = plt.subplots(2, figsize=(12, 8))
    #     timescale = np.arange(waveform.shape[0])
    #     axes[0].plot(timescale, waveform.numpy())
    #     axes[0].set_title('Waveform')
    #     axes[0].set_xlim([0, 16000])
    #     plot_spectrogram(spectrogram.numpy(), axes[1])
    #     axes[1].set_title('Spectrogram')
    #     plt.show()


    #     # Transform the waveform dataset to have spectrogram images and their corresponding labels as integer IDs
    #     def get_spectrogram_and_label_id(audio, label):
    #         #get_spectrogram()
    #         spectrogram = get_spectrogram(audio)

    #         # Adding a 1 at the end of the array (axis -1)
    #         spectrogram = tf.expand_dims(spectrogram, -1)
    #         # Not sure what is going on here??
    #         label_id = tf.argmax(label == commands)
    #         return spectrogram, label_id


    #     def spect_label_wrapper():
    #         spectrogram_ds = waveform_ds.map(
    #         get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)


    #         # Examine the spectrogram images from the spectrogram_ds for different samples (9 in this case)
    #         rows = 3
    #         cols = 3
    #         n = rows*cols
    #         fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    #         for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
    #             r = i // cols
    #             c = i % cols
    #             ax = axes[r][c]
    #             plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
    #             ax.set_title(commands[label_id.numpy()])
    #             ax.axis('off')

    #         plt.show()


    #     # Running the same preprocessing steps on the validation and test data
    #     def preprocess_dataset(files):
    #         files_ds = tf.data.Dataset.from_tensor_slices(files)
    #         output_ds = files_ds.map(get_waveform_and_label,
    #                                 num_parallel_calls=AUTOTUNE)
    #         output_ds = output_ds.map(
    #             get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
    #         return output_ds

    #     def preprocess_test_val():

    #         train_ds = spectrogram_ds
    #         val_ds = preprocess_dataset(val_files)
    #         test_ds = preprocess_dataset(test_files)

    #         # Divide each set of data into 64 batches
    #         batch_size = 64
    #         train_ds = train_ds.batch(batch_size)
    #         val_ds = val_ds.batch(batch_size)

    #         # Add dataset cache() and prefetch() operations to reduce read latency while training the model (????)
    #         train_ds = train_ds.cache().prefetch(AUTOTUNE)
    #         val_ds = val_ds.cache().prefetch(AUTOTUNE)

    #         # a cnn model is used for running convolutions on the images
    #         # Two additional layers are added
    #         # Resizing: downsamples the data to allow the model to train faster
    #         # Normalization: layer to normalize the value of each pixel in the image based on its mean and standard deviation

    #         for spectrogram, _ in spectrogram_ds.take(1):
    #             input_shape = spectrogram.shape
    #         print('Input shape:', input_shape)
    #         num_labels = len(commands)




if __name__=="__main__":
    
    path = 'data/mini_speech_commands'
    data = Data(path)
    data.random_seed()
    data.load_data_wrapper()


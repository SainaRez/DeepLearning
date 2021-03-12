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


# _DOWNLOAD_PATH = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
# _TEST_DOWNLOAD_PATH_ = 'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'

# _SPLITS = ['train', 'valid', 'test']

# WORDS = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
# SILENCE = '_silence_'
# UNKNOWN = '_unknown_'
# BACKGROUND_NOISE = '_background_noise_'
# SAMPLE_RATE = 16000

# class SpeechCommands(tfds.core.GeneratorBasedBuilder):
#   """The Speech Commands dataset for keyword detection."""

#     VERSION = tfds.core.Version('0.0.2')

#     def _info(self):
#         return tfds.core.DatasetInfo(
#             builder=self,
#             description=_DESCRIPTION,
#             # tfds.features.FeatureConnectors
#             features=tfds.features.FeaturesDict({
#                 'audio': tfds.features.Audio(
#                     file_format='wav', sample_rate=SAMPLE_RATE),
#                 'label': tfds.features.ClassLabel(names=WORDS + [SILENCE, UNKNOWN])
#             }),
#             supervised_keys=('audio', 'label')
#         )

#     def _split_generators(self, dl_manager):
#     """Returns SplitGenerators."""

#         dl_path, dl_test_path = dl_manager.download(
#             [_DOWNLOAD_PATH, _TEST_DOWNLOAD_PATH_])

#         train_paths, validation_paths = self._split_archive(
#             dl_manager.iter_archive(dl_path))

#         return [
#             tfds.core.SplitGenerator(
#                 name=tfds.Split.TRAIN,
#                 gen_kwargs={'archive': dl_manager.iter_archive(dl_path),
#                             'file_list': train_paths},
#             ),
#             tfds.core.SplitGenerator(
#                 name=tfds.Split.VALIDATION,
#                 gen_kwargs={'archive': dl_manager.iter_archive(dl_path),
#                             'file_list': validation_paths},
#             ),
#             tfds.core.SplitGenerator(
#                 name=tfds.Split.TEST,
#                 gen_kwargs={'archive': dl_manager.iter_archive(dl_test_path),
#                             'file_list': None},
#             ),
#         ]

#   def _generate_examples(self, archive, file_list):
#     """Yields examples."""
#     for path, file_obj in archive:
#         if file_list is not None and path not in file_list:
#         continue
#         relpath, wavname = os.path.split(path)
#         _, word = os.path.split(relpath)
#         example_id = '{}_{}'.format(word, wavname)
#         if word in WORDS:
#         label = word
#         elif word == SILENCE or word == BACKGROUND_NOISE:
#         # The main tar file already contains all of the test files, except for
#         # the silence ones. In fact it does not contain silence files at all.
#         # So for the test set we take the silence files from the test tar file,
#         # while for train and validation we build them from the
#         # _background_noise_ folder.
#         label = SILENCE
#         else:
#         # Note that in the train and validation there are a lot more _unknown_
#         # labels than any of the other ones.
#         label = UNKNOWN

#         if word == BACKGROUND_NOISE:
#         # Special handling of background noise. We need to cut these files to
#         # many small files with 1 seconds length, and transform it to silence.
#         audio_samples = np.array(
#             lazy_imports_lib.lazy_imports.pydub.AudioSegment.from_file(
#                 file_obj, format='wav').get_array_of_samples())

#         for start in range(0,
#                             len(audio_samples) - SAMPLE_RATE, SAMPLE_RATE // 2):
#             audio_segment = audio_samples[start:start + SAMPLE_RATE]
#             cur_id = '{}_{}'.format(example_id, start)
#             example = {'audio': audio_segment, 'label': label}
#             yield cur_id, example
#         else:
#         try:
#             example = {
#                 'audio':
#                     np.array(
#                         lazy_imports_lib.lazy_imports.pydub.AudioSegment
#                         .from_file(file_obj,
#                                     format='wav').get_array_of_samples()),
#                 'label':
#                     label,
#             }
#             yield example_id, example
#         except lazy_imports_lib.lazy_imports.pydub.exceptions.CouldntDecodeError:
#             pass

#   def _split_archive(self, train_archive):
#     train_paths = []
#     for path, file_obj in train_archive:
#         if 'testing_list.txt' in path:
#             train_test_paths = file_obj.read().strip().splitlines()
#             train_test_paths = [p.decode('ascii') for p in train_test_paths]
#         elif 'validation_list.txt' in path:
#             validation_paths = file_obj.read().strip().splitlines()
#             validation_paths = [p.decode('ascii') for p in validation_paths]
#         elif path.endswith('.wav'):
#             train_paths.append(path)

#     # Original validation files did include silence - we add them manually here
#     validation_paths.append(
#         os.path.join(BACKGROUND_NOISE, 'running_tap.wav'))

#     # The paths for the train set is just whichever paths that do not exist in
#     # either the test or validation splits.
#     train_paths = (
#         set(train_paths) - set(validation_paths) - set(train_test_paths))

#     return train_paths, validation_paths


class Data:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None


def load_data():
    data_dir = pathlib.Path('data/mini_speech_commands')
    if not data_dir.exists():
        tf.keras.utils.get_file(
        'mini_speech_commands.zip',
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir='.', cache_subdir='data')


class Train():
    def __init__(self):
    self.cnn

    def call_cnn():
        pass


# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


data_dir = pathlib.Path('data/mini_speech_commands')
if not data_dir.exists():
    tf.keras.utils.get_file(
        'mini_speech_commands.zip',
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir='.', cache_subdir='data')


# print the classes
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
print('Commands:', commands)


# printing some data about the data
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Number of examples per label:',
      len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
print('Example file tensor:', filenames[0])

# Dividing the data into train, eval and test
train_files = filenames[:6400]
val_files = filenames[6400: 6400 + 800]
test_files = filenames[-800:]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))

#


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

# Get the label from WAV file


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)

    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]

#  take in the filename of the WAV file and output a tuple containing the audio and labels for supervised training


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    print("shape of the waveform: ", waveform.shape)
    return waveform, label


# apply process_path to build training set to extract the audio-label pairs and check the results
AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
print("the datatset: ", files_ds)

# How is get_waveform_and_label gettign the file path??
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

# Examine a few of the data with their labels
rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
for i, (audio, label) in enumerate(waveform_ds.take(n)):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    ax.plot(audio.numpy())
    ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
    label = label.numpy().decode('utf-8')
    ax.set_title(label)

plt.show()


def add_padding_to_audio(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)

    # adding the padding at the end of the data
    equal_length = tf.concat([waveform, zero_padding], 0)
    return equal_length


# Converting the files to spectrogram, data shape = (124, 129)
def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)

    # adding the padding at the end of the data
    equal_length = tf.concat([waveform, zero_padding], 0)

    # stft splits the signal into windows of time and runs a Fourier transform on each window
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    # stft returns magnitude and phase and we only use magnitude (tf.abs)
    spectrogram = tf.abs(spectrogram)

    return spectrogram


for waveform, label in waveform_ds.take(10):
    label = label.numpy().decode('utf-8')
    spectrogram = get_spectrogram(waveform)

print('Label:', label)
print('Waveform shape:', waveform.shape)
print('Spectrogram shape:', spectrogram.shape)
print('Audio playback')
display.display(display.Audio(waveform, rate=16000))


# Plot the spectogram in the log scale and plot the wav file
def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    log_spec = np.log(spectrogram.T)
    #print("This is log spec: ", log_spec)
    # print("##############", log_spec.shape)   #(129, 124)
    height = log_spec.shape[0]    # 129
    X = np.arange(16000, step=height + 1)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16000])
plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.show()


# Transform the waveform dataset to have spectrogram images and their corresponding labels as integer IDs
def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)

    # Adding a 1 at the end of the array (axis -1)
    spectrogram = tf.expand_dims(spectrogram, -1)
    # Not sure what is going on here??
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)


# Examine the spectrogram images from the spectrogram_ds for different samples (9 in this case)
rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
    ax.set_title(commands[label_id.numpy()])
    ax.axis('off')

plt.show()


# Build and Train the model


# Running the same preprocessing steps on the validation and test data
def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label,
                             num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
    return output_ds


train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

# Divide each set of data into 64 batches
batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

# Add dataset cache() and prefetch() operations to reduce read latency while training the model (????)
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# a cnn model is used for running convolutions on the images
# Two additional layers are added
# Resizing: downsamples the data to allow the model to train faster
# Normalization: layer to normalize the value of each pixel in the image based on its mean and standard deviation

for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = len(commands)


# The adapt method, when called on the training data, calculates mean and standard deviation
norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

# Add all the layers
input_ = layers.Input(shape=input_shape)  # (129, 124)
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
output_ = layers.Dense(num_labels)(dropout_2)


# build the model
model = tf.keras.Model(inputs=input_, outputs=output_)
model.summary()


# Compile and fit the model
model.compile(optimizer=tf.keras.optimizers.Adam(
), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

EPOCHS = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2))

# Visualizing the data
# The loss curve for the training and validation data

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()


# Now running the model on the test set and checking the result
test_audio = []
test_labels = []

for audio, label in test_ds:
    test_audio.append(audio.numpy())
    test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

pred = model.predict(test_audio)
y_pred = np.argmax(pred, axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')


# Confusion matrix
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

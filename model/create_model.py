import os
import glob
import librosa
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import matplotlib.pyplot as pl
from tensorflow import keras
from tensorflow.python.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
from keras.models import load_model

max_pad_len = 7000


def extract_feature(file_name):
    print(‘file
    name:’, file_name)
    try:
        audio, sample_rate = librosa.load(file_name, res_type=’kaiser_fast’)
        mfccs = librosa.feature.mfcc(y=audio, sr=9600, n_mfcc=40)
        pad_width = max_pad_len – mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode=’constant’)
        print(mfccs.shape)
        print(audio)
        print(“original
        sr: ”, sample_rate)
        print(“new
        sr: 9600”)

        except Exception as e:
        print(“Error
        encountered
        while parsing file:”, file_name)
        print(e)
        return None
    return mfccs

fulldatasetpath = r’D: / dataset_final / audioset / final / final’
metadata = pd.read_csv(r’D: / dataset_final / csvfile / final_csv_cut.csv’)
features = []
for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath(fulldatasetpath), str(row[“slice_file_name”]))
    class_label = row[“classID”]
    data = extract_feature(file_name)
    features.append([data, class_label])
featuresdf = pd.DataFrame(features, columns=[‘feature’, ‘class_label’])

x = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))
x_train, x_test, y_train, y_test = train_test_split(x, yy, test_size=0.2, random_state=42)
num_rows = 40
num_columns = 7000
num_channels = 1
print("train data shape")
print(x_train.shape)
print(x_test.shape)
x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)
print("\ntrain data reshape 결과")
print(x_train.shape)
print(x_test.shape)

num_labels = yy.shape[1]
filter_size = 2
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 2, input_shape = (num_rows, num_columns, num_channels), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 32, kernel_size = 2, activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 64, kernel_size = 2, activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 128, kernel_size = 2, activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())
model.add(Dense(num_labels, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy'
                , metrics = ['accuracy']
                , optimizer = 'adam')

model.summary()
score = model.evaluate(x_test, y_test, verbose = 1)
accuracy = 100 * score[1]
print('Pre-training accuracy: %.4f%%' % accuracy)

history = model.fit(x_train, y_train, batch_size=256, epochs=150, validation_data=(x_test, y_test))

model.save('model_final_test_150.h5')

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title('Model ' + metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

plot_graphs(history, 'loss')
plot_graphs(history, 'accuracy')

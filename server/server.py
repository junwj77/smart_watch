import serial
import numpy as np
import time
import librosa
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from python_speech_features import mfcc
from scipy.fft import ifft

ser = serial.Serial('COM7', 115200)
ser.timeout = 1

duration = 7
fft_data = []
start_time = time.time()
while time.time() - start_time < duration:
    line = ser.readline().strip()
    fft_data_str = line.decode()
    fft_data_list = fft_data_str.split()
    fft_data += [float(value) for value in fft_data_list]
ser.close()

fft_data = np.array(fft_data_list, dtype=np.float)
time_domain_signal = np.fft.ifft(fft_data)
print(time_domain_signal)

sample_rate = 9600
num_samples = sample_rate * duration
time_domain_signal = time_domain_signal[:num_samples]
n_mfcc = 40
mfccs = librosa.feature.mfcc(y=time_domain_signal.real, sr=sample_rate, n_mfcc=n_mfcc)
print(mfccs.shape)

max_pad_len = 7000
pad_width = max_pad_len - mfccs.shape[1]
mfccs = np.pad(mfccs, pad_width=((0,0), (0, pad_width)), mode='constant')
print(mfccs.shape)

model_path = "model_final_test_150.h5"
le = LabelEncoder()
# Load the model
model = load_model(model_path)
classID = [1, 2, 3, 4]
le.fit(classID)
prediction_feature = mfccs
prediction_feature = prediction_feature.reshape(1, 40, 7000, 1)
predicted_vector = model.predict(prediction_feature)
predicted_class = np.argmax(predicted_vector[0])
max_probability = np.max(predicted_vector[0])

if max_probability >= 0.9:
    predicted_class += 1
else:
    predicted_class = 0
if predicted_class == 0:
    print("The sound does not belong to any class.")
else:
    print("The predicted class is:", predicted_class)
for i, proba in enumerate(predicted_vector[0]):
    category = le.inverse_transform([i])
    print(category[0], "\t\t : ", format(proba, '.32f'))

print("Start")
import serial
import time
print("Start")
port="COM6".
bluetooth=serial.Serial(port, 9600)
print("Connected")
bluetooth.flushInput()
bluetooth.write(str.encode(str(predicted_class)))
print("Done")

from IPython.display import display, Javascript

def restart_kernel_and_run_all_cells():
    display(Javascript('''
        IPython.notebook.kernel.restart();
        IPython.notebook.execute_all_cells();
    '''))

restart_kernel_and_run_all_cells()

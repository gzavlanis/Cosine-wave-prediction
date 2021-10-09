import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, LeakyReLU
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
import random
from keras.utils.vis_utils import plot_model

t= np.linspace(1, 300, 3000)
x_volts= 10* np.cos(t/(2*np.pi)) # signal generation
print(x_volts)
plt.subplot(3, 1, 1)
plt.plot(t, x_volts)
plt.title('Signal')
plt.ylabel('Voltage (Volts)')
plt.xlabel('Time (seconds)')

x_watts= x_volts** 2
plt.subplot(3, 1, 2)
plt.plot(t, x_watts)
plt.title('Signal Power')
plt.ylabel('Power (Watts)')
plt.xlabel('Time (seconds)')

x_db= 10* np.log(x_watts)
plt.subplot(3, 1, 3)
plt.plot(t, x_db)
plt.title('Signal Power in dB')
plt.ylabel('Power (dB)')
plt.xlabel('Time (s)')
plt.show()

# Adding noise using target SNR
target_snr_db= 20 # set a target SNR
sig_avg_watts= np.mean(x_watts) # Calculate signal power and convert to dB 
sig_avg_db= 10* np.log10(sig_avg_watts)
noise_avg_db= sig_avg_db- target_snr_db # Calculate noise according to [2] then convert to watts
noise_avg_watts= 10** (noise_avg_db/ 10)
mean_noise= 0 # Generate a sample of white noise
noise_volts= np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
y_volts= x_volts+ noise_volts # Noise up the original signal
print(y_volts)

# Plot signal with noise
plt.subplot(2, 1, 1)
plt.plot(t, y_volts)
plt.title('Signal with noise')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
y_watts= y_volts**2
y_db= 10* np.log10(y_watts)
plt.subplot(2, 1, 2)
plt.plot(t, 10* np.log10(y_volts** 2))
plt.title('Signal with noise (dB)')
plt.ylabel('Power (dB)')
plt.xlabel('Time (s)')
plt.show()

# Adding noise using a target noise power
target_noise_db= 10 # Set a target channel noise power to something very noisy
target_noise_watts= 10** (target_noise_db/ 10) # Convert to linear Watt units
mean_noise= 0 # Generate noise samples
noise_volts1= np.random.normal(mean_noise, np.sqrt(target_noise_watts), len(x_watts))
y_volts1= x_volts+ noise_volts1 # Noise up the original signal (again) and plot
print(y_volts)

# Plot signal with noise
plt.subplot(2, 1, 1)
plt.plot(t, y_volts1)
plt.title('Signal with noise')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
y_watts1= y_volts1** 2
y_db= 10* np.log10(y_watts1)
plt.subplot(2, 1, 2)
plt.plot(t, 10* np.log10(y_volts1** 2))
plt.title('Signal with noise')
plt.ylabel('Power (dB)')
plt.xlabel('Time (s)')
plt.show()

n_timesteps= 3 #define the timesteps of the problem
# Create the deep BiLSTM network and make a figure of it
model= Sequential()
model.add(Bidirectional(LSTM(32, return_sequences= True), input_shape=(n_timesteps, 1)))
model.add(TimeDistributed(Dense(128, activation= LeakyReLU(alpha= 0.01))))
model.add(Dense(128, activation= LeakyReLU(alpha= 0.01)))
model.add(Dense(64, activation= LeakyReLU(alpha= 0.01)))
model.add(Dense(1, activation= 'linear'))
model.compile(loss= MeanSquaredError(), optimizer= Adam(learning_rate=0.01), metrics=[ 'accuracy' ])
print(model.summary())
plot_model(model, to_file='BiLSTM.png', show_shapes=True, show_layer_names=True)

X= np.array([x_volts]) # input training signal
Y= np.array([y_volts]) # output training signal
print(X)
print(Y)

# convert the arrays to tensors
X= tf.convert_to_tensor(X, dtype= tf.float32)
Y= tf.convert_to_tensor(Y, dtype= tf.float32)
print(X)
print(Y)

# reshape data for entering the model
X= tf.reshape(X, [1000, n_timesteps, 1])
Y= tf.reshape(Y, [1000, n_timesteps, 1])
print(X)
print(Y)

# train the model and plot results for loss
def train_model(model, n_timesteps):
    hist= model.fit(X, Y, epochs= 20, batch_size= 64)
    loss= hist.history['loss']
    return loss

loss= train_model(model, n_timesteps)
plt.plot(loss, label= 'Loss')
plt.title('Training loss of the model')
plt.xlabel('epochs', fontsize= 18)
plt.ylabel('loss', fontsize= 18)
plt.grid()
plt.legend()
plt.show()

# predict the output signal Y
Yhat= model.predict(Y, verbose= 0)
Yhat= Yhat.reshape(3000, 1)
print(Yhat)

# predict the signal without noise X
Xhat= model.predict(X, verbose= 0)
Xhat= Xhat.reshape(3000, 1)
print(Xhat)

# Reshape X and Y back to normal shape
X= tf.reshape(X, [3000, 1])
Y= tf.reshape(Y, [3000, 1])

# Plot real waveform and predicted waveform for Y
plt.plot(t, Y, 'r', label= 'Real waveform')
plt.plot(t, Yhat, 'b', label= 'Predicted waveform')
plt.title('Plot of real vs predicted waveforms', fontsize= 16)
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid()
plt.show()

# Plot real waveform and predicted waveform for X
plt.plot(t, X, 'r', label= 'Real waveform')
plt.plot(t, Xhat, 'b', label= 'Predicted waveform')
plt.title('Plot of real vs predicted waveforms', fontsize= 16)
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid()
plt.show()

# Predict a more noisy signal without training
Y1= np.array([y_volts1])
Y1= tf.convert_to_tensor(Y1, dtype= tf.float32)
Y1= tf.reshape(Y1, [1000, n_timesteps, 1])
Yhat1= model.predict(Y1, verbose= 0)
Yhat1= Yhat1.reshape(3000, 1)
print(Yhat1)
Y1= tf.reshape(Y1, [3000, 1])
pyplot.plot(t, Y1, 'r', label= 'Real waveform')
pyplot.plot(t, Yhat1, 'b', label= 'Predicted waveform')
pyplot.title('Prediction of a more noisy wave without training', fontsize= 18)
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
pyplot.legend()
pyplot.grid()
pyplot.show()

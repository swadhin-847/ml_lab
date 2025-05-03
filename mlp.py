import os
import tensorflow as tf
import csv
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.get_logger().setLevel('ERROR')

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def read_csv_file(filename):
    passengers = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            passengers.append(int(row[1]))
    return passengers

# Parameters
n_steps = 3
epochs = 2000
batch_size = 32
learning_rate = 0.001
neurons = 100

# Data loading
raw_seq = read_csv_file('/home/swadhin/Desktop/ml_lab/airline-passengers.csv')

# Prepare data
X, y = split_sequence(raw_seq, n_steps)

# Model definition
model = Sequential()
model.add(Input(shape=(n_steps,)))
model.add(Dense(neurons, activation='relu'))
model.add(Dense(1))

# Compile model
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')

# Train model
model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

# Predict
x_input = array([136, 148, 148])
x_input = x_input.reshape((1, n_steps))
yhat = model.predict(x_input, verbose=0)
print(f"Predicted next value: {yhat}")


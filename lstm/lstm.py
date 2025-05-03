import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras_tuner import Hyperband

# Define the generate_time_series_data function
def generate_time_series_data(seq_length, num_samples):
    X = np.random.randn(num_samples, seq_length)  # Random features
    y = np.random.randn(num_samples)  # Random target values
    return X, y

# Generate dummy time series data
sequence_length = 50
num_samples = 1000
X, y = generate_time_series_data(seq_length=sequence_length, num_samples=num_samples)

# Reshape X for LSTM [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Scale data using MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
y = scaler.fit_transform(y.reshape(-1, 1))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model-building function for hyperparameter tuning
def build_model(hp):
    model = Sequential()
    model.add(
        LSTM(
            units=hp.Int('units', min_value=32, max_value=128, step=16),
            activation='relu',
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )
    model.add(Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='mse'
    )
    return model

# Hyperparameter tuning
hypermodel = Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=20,
    factor=3,
    directory='my_dir',
    project_name='lstm_tuning'
)

# Search for the best hyperparameters
hypermodel.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32, verbose=1)

# Get the best hyperparameters and model
best_hps = hypermodel.get_best_hyperparameters(num_trials=1)[0]
print(f"Best units: {best_hps.get('units')}")
print(f"Best learning rate: {best_hps.get('learning_rate')}")

# Build the model with the best hyperparameters
best_model = hypermodel.hypermodel.build(best_hps)

# Train the model with the optimal hyperparameters
history = best_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=32, verbose=1)

# Evaluate the model
loss = best_model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss}')

# Make predictions
y_pred = best_model.predict(X_test)

# Rescale predictions and actual values to original scale
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# Plot predictions vs actual values
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='Actual')
plt.plot(y_pred_rescaled, label='Predicted')
plt.legend()
plt.title('LSTM Predictions vs Actual Values')
plt.show()


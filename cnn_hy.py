# Import necessary libraries
import optuna  # Ensure Optuna is installed with `pip install optuna`
import tensorflow as tf
import mnist
import Sequential
import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import Adam, SGD
import to_categorical
from sklearn.model_selection import train_test_split

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Split training data for validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Objective function for Optuna
def objective(trial):
    # Hyperparameters to optimize
    num_filters = trial.suggest_categorical("num_filters", [16, 32, 64])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5])
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    
    # Build CNN model
    model = Sequential([
        Conv2D(num_filters, kernel_size=(kernel_size, kernel_size), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate) if optimizer_name == "adam" else SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Train model
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, batch_size=128, verbose=0)
    
    # Return validation accuracy
    val_accuracy = history.history["val_accuracy"][-1]
    return val_accuracy

# Optimize using Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)

# Evaluate the best model on the test set
best_params = study.best_params
num_filters = best_params["num_filters"]
kernel_size = best_params["kernel_size"]
dropout_rate = best_params["dropout_rate"]
learning_rate = best_params["learning_rate"]
optimizer_name = best_params["optimizer"]

# Build the best model
best_model = Sequential([
    Conv2D(num_filters, kernel_size=(kernel_size, kernel_size), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(dropout_rate),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(dropout_rate),
    Dense(10, activation='softmax')
])

optimizer = Adam(learning_rate=learning_rate) if optimizer_name == "adam" else SGD(learning_rate=learning_rate)
best_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Train and evaluate the best model
best_model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=1)
test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")


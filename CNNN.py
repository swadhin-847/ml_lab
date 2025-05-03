import tensorflow as tf
from keras import layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define the CNN model for blurring
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(None, None, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Function to preprocess the input image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))  # Resize to a fixed size
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to display images
def display_images(original, modified):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original.squeeze(), cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Blurred Image')
    plt.imshow(modified.squeeze(), cmap='gray')

    plt.show()

# Dummy dataset for training (Gaussian blur simulation)
def generate_training_data():
    x_train = []
    y_train = []
    for _ in range(500):
        img = np.random.rand(128, 128)  # Generate random noise image
        blurred = cv2.GaussianBlur(img, (5, 5), 0)  # Apply Gaussian blur

        x_train.append(np.expand_dims(img, axis=-1))
        y_train.append(np.expand_dims(blurred, axis=-1))

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train, y_train

# Generate training data
x_train, y_train = generate_training_data()

# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=16, validation_split=0.2)

# Visualize training progress
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy During Training')
plt.legend()

plt.show()

# Load and preprocess the input image
input_image_path = '/home/swadhin/Desktop/ml_lab/input_image.jpg'  # Replace with the path to your image
input_image = preprocess_image(input_image_path)

# Predict the blurred image
blurred_image = model.predict(input_image)

# Display the original and blurred images
display_images(input_image, blurred_image)

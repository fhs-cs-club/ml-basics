import tensorflow as tf
from tensorflow.keras import layers, models

# Define a simple CNN model
model = models.Sequential([
    # Convolutional layer with 32 filters, each of size 3x3, and ReLU activation
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Max-pooling layer with a 2x2 window
    layers.MaxPooling2D((2, 2)),
    # Another convolutional layer with 64 filters, each of size 3x3, and ReLU activation
    layers.Conv2D(64, (3, 3), activation='relu'),
    # Max-pooling layer
    layers.MaxPooling2D((2, 2)),
    # Flatten the output for the fully connected layers
    layers.Flatten(),
    # Fully connected layer with 64 units and ReLU activation
    layers.Dense(64, activation='relu'),
    # Output layer with 10 units (assuming you're working with a 10-class classification problem)
    layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()
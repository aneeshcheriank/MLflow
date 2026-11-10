import mlflow
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Enable autologging
mlflow.tensorflow.autolog()

# Prepare sample data
data = np.random.uniform(size=[20, 28, 28, 3])
label = np.random.randint(2, size=20)

# Define model
model = keras.Sequential([
    keras.Input([28, 28, 3]),
    keras.layers.Conv2D(8, 2),
    keras.layers.MaxPool2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(2),
    keras.layers.Softmax(),
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(0.001),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Training with automatic logging
with mlflow.start_run():
    model.fit(data, label, batch_size=5, epochs=2)
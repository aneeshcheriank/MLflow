import mlflow
import numpy as np
import tensorflow as tf
from tensorflow import keras

import click

@click.command()
@click.option("--channels", default=8, help="Number of channels in the input data.")
@click.option("--window", default=3, help="convolution window size.")
@click.option("--lr", default=0.001, help="Learning rate for the optimizer.")
def train_model(channels, window, lr):
    # Prepare sample data
    data = np.random.uniform(size=[20, 28, 28, 3])
    label = np.random.randint(2, size=20)

    # Define model
    model = keras.Sequential([
        keras.Input([28, 28, 3]),
        keras.layers.Conv2D(channels, window),
        keras.layers.MaxPool2D(window),
        keras.layers.Flatten(),
        keras.layers.Dense(2),
        keras.layers.Softmax(),
    ])
    
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(lr),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(data, label, batch_size=5, epochs=2)

    
if __name__ == "__main__":

    # Enable autologging
    mlflow.tensorflow.autolog()
    train_model()
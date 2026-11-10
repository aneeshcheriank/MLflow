# Hyper parameter optimization using MLflow and TensorFlow
# Optuna

import mlflow
import tensorflow as tf
from tensorflow import keras
import optuna

import numpy as np
from sklearn.model_selection import train_test_split


def objective(trial, x_train, y_train, x_val, y_val):
    """Optuna objective for TensorFlow hyperparameter tuning."""
    with mlflow.start_run(nested=True):
        # Define hyperparameter search space
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "units": trial.suggest_int("units", 32, 512),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        }

        # Create model with hyperparameters
        model = keras.Sequential([
            keras.layers.Input(shape=(28, 28, 3)),
            keras.layers.Flatten(),
            keras.layers.Dense(params["units"], activation="relu"),
            keras.layers.Dropout(params["dropout"]),
            keras.layers.Dense(10, activation="softmax"),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train and evaluate
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, verbose=0)

        val_accuracy = max(history.history["val_accuracy"])
        mlflow.log_metric("val_accuracy", val_accuracy)

        return val_accuracy

# Prepare sample data
data = np.random.uniform(size=[100, 28, 28, 3])
label = np.random.randint(2, size=100)

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(data, label, test_size=0.2, random_state=42)

# Main experiment run
with mlflow.start_run(run_name="tensorflow_hyperparameter_optimization"):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, x_train, y_train, x_val, y_val), n_trials=20)

    # Log best parameters and results
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_val_accuracy", study.best_value)
# steps/training.py
from zenml import step, log_model_metadata, ArtifactConfig

from zenml.client import Client
import numpy as np
import mlflow
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Dropout,
    BatchNormalization,
    GRU,
    Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    LearningRateScheduler,
)
from zenml.logger import get_logger
from zenml import ArtifactConfig, log_artifact_metadata, step
from typing import Dict
from typing_extensions import Annotated

experiment_tracker = Client().active_stack.experiment_tracker

# Import the KerasFunctionalMaterializer and register it with the Model class
from tensorflow import keras
from steps.keras_materializer import KerasMaterializer


# Initialize logger
logger = get_logger(__name__)


# Define a function to build the model with a default input shape
def build_model(input_shape):
    # Define the input layer with the given shape
    inputs = Input(shape=input_shape)

    # First convolutional layer with 64 filters and kernel size of 3
    x = Conv1D(64, kernel_size=3, activation="relu")(inputs)
    # Batch normalization to maintain the mean output close to 0 and the output standard deviation close to 1
    x = BatchNormalization()(x)
    # Max pooling to reduce the dimensionality of the input volume
    x = MaxPooling1D(pool_size=2)(x)
    # Dropout for regularization to reduce overfitting
    x = Dropout(0.3)(x)

    # Second convolutional layer with 128 filters and kernel size of 3
    x = Conv1D(128, kernel_size=3, activation="relu")(x)
    # Batch normalization
    x = BatchNormalization()(x)
    # Max pooling
    x = MaxPooling1D(pool_size=2)(x)
    # Dropout
    x = Dropout(0.4)(x)

    # Third convolutional layer with 256 filters and kernel size of 3
    x = Conv1D(256, kernel_size=3, activation="relu")(x)
    # Batch normalization
    x = BatchNormalization()(x)
    # Max pooling
    x = MaxPooling1D(pool_size=2)(x)
    # Dropout
    x = Dropout(0.4)(x)

    # First GRU layer with 128 units, returning sequences to allow the next GRU layer to have sequential input
    x = GRU(128, return_sequences=True)(x)
    # Dropout
    x = Dropout(0.5)(x)
    # Second GRU layer with 64 units
    x = GRU(64)(x)
    # Dropout
    x = Dropout(0.5)(x)

    # Dense layer with 64 units and ReLU activation, including L2 regularization
    x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
    # Dropout
    x = Dropout(0.5)(x)
    # Another dense layer with 32 units and ReLU activation, including L2 regularization
    x = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(x)
    # Output layer with a single unit and sigmoid activation for binary classification
    outputs = Dense(1, activation="sigmoid")(x)

    # Create the model with specified inputs and outputs
    model = Model(inputs=inputs, outputs=outputs)
    return model


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 5:
        lr *= 0.5
    if epoch > 20:
        lr *= 0.5
    if epoch > 30:
        lr *= 0.5
    return lr


early_stopping = EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6)
lr_scheduler = LearningRateScheduler(lr_schedule)


@step(
    experiment_tracker=experiment_tracker.name,
    enable_cache=True,
    output_materializers=KerasMaterializer,
)
def training_step(
    split_data: Dict[str, np.ndarray], epochs: int = 25, batch_size: int = 32
) -> Annotated[
    keras.Model,
    ArtifactConfig(
        name="trigger_word_detection_model",
        is_model_artifact=True,
    ),
]:
    """
    Executes the training step for a trigger word detection model.

    This function takes split data (training, validation, and test sets), the number of epochs, and the batch size as inputs.
    It first checks for GPU availability and then proceeds to build and compile the model using the Adam optimizer and binary crossentropy loss.
    The model is trained on the training data and validated on the validation data. Additionally, this function logs the training parameters (epochs and batch size) using MLflow.

    Parameters:
    - split_data (Dict[str, np.ndarray]): A dictionary containing the training, validation, and test data with keys 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'.
    - epochs (int, optional): The number of epochs to train the model. Defaults to 25.
    - batch_size (int, optional): The size of the batches of data. Defaults to 32.

    Returns:
    - Annotated[Model, ArtifactConfig]: The trained model annotated with ArtifactConfig, indicating it is a model artifact named "trigger_word_detection_model".

    Note:
    - The function verifies the availability of a GPU and prefers using it over CPU for training.
    - It uses MLflow for logging the training parameters and TensorFlow's autolog feature for automatic logging.
    """

    # Verify TensorFlow can see the GPU
    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )
    if tf.config.experimental.list_physical_devices("GPU"):
        print("Using GPU")
    else:
        print("Using CPU")
    X_train = split_data["X_train"]
    X_val = split_data["X_val"]
    X_test = split_data["X_test"]
    y_train = split_data["y_train"]
    y_val = split_data["y_val"]
    y_test = split_data["y_test"]

    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    mlflow.tensorflow.autolog()

    # Log parameters
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)

    # Add the callbacks
    callbacks = [early_stopping, reduce_lr, lr_scheduler]

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    # Log metrics
    for epoch in range(epochs):
        mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
        mlflow.log_metric(
            "train_accuracy", history.history["accuracy"][epoch], step=epoch
        )
        mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
        mlflow.log_metric(
            "val_accuracy", history.history["val_accuracy"][epoch], step=epoch
        )
    print(
        "Train accuracy: {}, Val Accuracy: {}".format(
            history.history["accuracy"][-1], history.history["val_accuracy"][-1]
        )
    )
    log_artifact_metadata(
        metadata={
            "train_accuracy": history.history["accuracy"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],
            "train loss": history.history["loss"][-1],
            "val loss": history.history["val_loss"][-1],
            "epochs": epochs,
            "batch_size": batch_size,
        },
        artifact_name="trigger_word_detection_model",
    )

    log_model_metadata(
        metadata={
            "train_accuracy": history.history["accuracy"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],
            "train loss": history.history["loss"][-1],
            "val loss": history.history["val_loss"][-1],
            "epochs": epochs,
            "batch_size": batch_size,
        }
    )
    mlflow.tensorflow.log_model(model, "trigger_word_detection_model")
    # save_model(model, "saved_models/model.keras")
    model.save("saved_models/trigger_word_detection_model.keras", save_format="keras")
    return model

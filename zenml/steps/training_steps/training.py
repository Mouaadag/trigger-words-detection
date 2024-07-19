# steps/training.py
from zenml import step, log_model_metadata, ArtifactConfig, get_step_context
from zenml.model_registries.base_model_registry import RegistryModelVersion
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

# from tensorflow.keras.optimizers.legacy import Adam
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

# Import the KerasFunctionalMaterializer and register it with the Model class
from tensorflow import keras
from zenml.integrations.tensorflow.materializers.keras_materializer import (
    KerasMaterializer,
)
from zenml.integrations.mlflow.steps.mlflow_registry import (
    mlflow_register_model_step,
)
from typing_extensions import Annotated
import os
import shutil
import mlflow.tensorflow

# Initialize logger
logger = get_logger(__name__)
experiment_tracker = Client().active_stack.experiment_tracker


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
    enable_cache=False,
    output_materializers=KerasMaterializer,
)
def training_step(
    split_data: Dict[str, np.ndarray],
    epochs: int = 25,
    batch_size: int = 32,
    name: str = "trigger_word_detection_model",
) -> Annotated[
    keras.Model,
    ArtifactConfig(
        name="trigger_word_detection_model",
        is_model_artifact=True,
    ),
]:
    """
    This function initializes the model training process with the given dataset, epochs, batch size, and model name.
    It first checks for GPU availability to leverage hardware acceleration.
    If a GPU is available, it proceeds with GPU-based training; otherwise, it defaults to CPU training.
    The model is then built with a specified input shape derived from the training data,
    compiled with the Adam optimizer with a learning rate of 1e-3 and a clipnorm of 1.0,
    and finally trained with the provided training and validation datasets.

    Parameters:
        split_data (Dict[str, np.ndarray]): A dictionary containing the training, validation, and test datasets.
        Expected keys are 'X_train', 'X_val', 'y_train', 'y_val'.
        epochs (int, optional): The number of epochs for which the model should be trained. Defaults to 25.
        batch_size (int, optional): The size of the batches of data during training. Defaults to 32.
        name (str, optional): The name of the model to be used for saving and logging purposes. Defaults to "trigger_word_detection_model".

    Returns:
        Annotated[keras.Model, ArtifactConfig]: The trained Keras model,
        annotated with ArtifactConfig to indicate it is a model artifact with a specified name and flag indicating it is a model artifact.

    Note:
        The function prints the number of available GPUs and indicates whether the training will proceed on GPU or CPU.
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
    y_train = split_data["y_train"]
    y_val = split_data["y_val"]

    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.compile(
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    mlflow.tensorflow.autolog(registered_model_name=name)

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
        artifact_name=name,
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
    model_save_path = os.path.join("saved_models", name)
    if os.path.exists(model_save_path):
        shutil.rmtree(model_save_path)
    os.makedirs(model_save_path, exist_ok=True)
    # mlflow.tensorflow.save_model(model, model_save_path)
    model.save("saved_models/trigger_word_detection_model.keras")
    # log model with mlflow
    mlflow.tensorflow.log_model(model, name)
    # register mlflow model
    mlflow_register_model_step.entrypoint(
        model,
        name=name,
    )
    # keep track of mlflow version for future use
    model_registry = Client().active_stack.model_registry
    if model_registry:
        version = model_registry.get_latest_model_version(name=name, stage=None)
        if version:
            model_ = get_step_context().model
            model_.log_metadata({"model_registry_version": version.version})
    return model

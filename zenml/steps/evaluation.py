# steps/evaluation.py
from zenml import step, log_artifact_metadata
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import tensorflow as tf
from typing import Dict
from tensorflow.keras.models import load_model, Model
from zenml.client import Client
from zenml.logger import get_logger
from zenml.logger import get_logger
from tensorflow import keras
from steps.keras_materializer import KerasMaterializer


# class KerasModelMaterializer(BaseMaterializer):
#     ASSOCIATED_TYPES = (Model,)

#     def load(self, data_type: Type[Model]) -> Model:
#         return load_model(f"{self.uri}.keras")

#     def save(self, model: Model) -> None:
#         model.save(f"{self.uri}.keras")  # Adding .keras extension


# # Assuming serialize_model is intended for manual serialization handling
# def serialize_model(model: Model, path: str) -> None:
#     materializer = KerasModelMaterializer()
#     materializer.uri = path
#     materializer.save(model)


# def deserialize_model(path: str) -> Model:
#     materializer = KerasModelMaterializer()
#     materializer.uri = path
#     return materializer.load(Model)


experiment_tracker = Client().active_stack.experiment_tracker

# Initialize logger
logger = get_logger(__name__)


@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def evaluate_model(
    model: keras.Model, split_data: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    # Assuming model is already loaded and passed as an argument
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    # evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)

    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.round(y_pred).flatten()

    precision = precision_score(y_test, y_pred_classes)
    recall = recall_score(y_test, y_pred_classes)
    f1 = f1_score(y_test, y_pred_classes)

    # Log metrics to MLflow
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)

    log_artifact_metadata(
        metadata={
            "test_accuracy": accuracy,
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1_score": float(f1),
        },
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

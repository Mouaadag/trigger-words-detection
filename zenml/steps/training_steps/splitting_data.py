# steps/splitting_data.py
from zenml import step, log_artifact_metadata
from sklearn.model_selection import train_test_split
import numpy as np
import mlflow
import logging
from typing import Dict
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker
from zenml.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


@step(experiment_tracker=experiment_tracker.name, enable_cache=True)
# @step(step_operator="azure_step_op")
def split_dataset(
    data: Dict[str, np.ndarray],
    test_size: float = 0.01,
    val_size: float = 0.01,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Splits the dataset into training, validation, and test sets.

    This function takes a dictionary containing the dataset, a test size, a validation size, and a random state as inputs.

    It performs two splits: first, it separates a test set from the rest of the data; then, it splits the remaining data into training and validation sets.

    The function also logs the shapes of the resulting datasets using MLflow.

    Parameters:
    - data (Dict[str, np.ndarray]): A dictionary with keys 'X' and 'y' representing the features and labels of the dataset, respectively.
    - test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.01.
    - val_size (float, optional): The proportion of the remaining data (after removing the test set) to include in the validation split. Defaults to 0.01.
    - random_state (int, optional): A seed used by the random number generator for reproducibility. Defaults to 42.

    Returns:
    - Dict[str, np.ndarray]: A dictionary containing the split datasets with keys 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', and 'y_test'.

    The function logs the shapes of 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', and 'y_test' using MLflow.

    It is decorated with `@step` from ZenML, indicating it's a step in a ZenML pipeline, with caching enabled and associated with a specific experiment tracker.
    """

    logging.info("Executing split_dataset step")

    X, y = data["X"], data["y"]

    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: separate validation set from training set
    # val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )
    # Start a new MLflow run
    # Assuming X_train, X_val, X_test, y_train, y_val, and y_test are already defined
    mlflow.log_param("X_train_shape", str(X_train.shape))
    mlflow.log_param("X_val_shape", str(X_val.shape))
    mlflow.log_param("X_test_shape", str(X_test.shape))
    mlflow.log_param("y_train_shape", str(y_train.shape))
    mlflow.log_param("y_val_shape", str(y_val.shape))
    mlflow.log_param("y_test_shape", str(y_test.shape))

    # End the run explicitly (if not using the 'with' context manager)

    # Log metadata

    log_artifact_metadata(
        {
            "X_train_shape": str(X_train.shape),
            "X_val_shape": str(X_val.shape),
            "X_test_shape": str(X_test.shape),
            "y_train_shape": str(y_train.shape),
            "y_val_shape": str(y_val.shape),
            "y_test_shape": str(y_test.shape),
        }
    )
    print("Metadata logged")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

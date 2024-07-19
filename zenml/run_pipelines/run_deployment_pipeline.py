from pipelines.run_deployment import deployment_pipeline
from zenml.client import Client
import mlflow
from zenml.logger import get_logger

logger = get_logger(__name__)
mlflow.set_experiment("trigger_detection_word_tracker")


if __name__ == "__main__":
    experiment_tracker = Client().active_stack.experiment_tracker
    print(f"Experiment tracking URI: {experiment_tracker.get_tracking_uri()}")
    logger.info(f"Experiment tracking URI: {experiment_tracker.get_tracking_uri()}")
    # Set the experiment name

    print(f"Experiment tracking URI: {experiment_tracker.get_tracking_uri()}")
    logger.info(f"Experiment tracking URI: {experiment_tracker.get_tracking_uri()}")
    # Set the experiment name
    deployment_pipeline()

    print("To view the MLflow UI, run:")
    print(f"mlflow ui --backend-store-uri {experiment_tracker.get_tracking_uri()}")

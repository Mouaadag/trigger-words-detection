from pipelines.run_training import train_and_deploy_pipeline
from zenml.client import Client
import mlflow
from zenml.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    positive_directory = "data/positives_samples"
    negative_directory = "data/negatives_samples"

    experiment_tracker = Client().active_stack.experiment_tracker
    print(f"Experiment tracking URI: {experiment_tracker.get_tracking_uri()}")
    logger.info(f"Experiment tracking URI: {experiment_tracker.get_tracking_uri()}")
    # Set the experiment name
    mlflow.set_experiment("trigger_detection_word_tracker")

    train_and_deploy_pipeline(
        positive_dir=positive_directory,
        negative_dir=negative_directory,
        epochs=25,
        batch_size=32,
    )

    print("To view the MLflow UI, run:")
    print(f"mlflow ui --backend-store-uri {experiment_tracker.get_tracking_uri()}")
# mlflow ui --backend-store-uri file:/Users/mouaad/DL_algo/trigger-words-detection/zenml/artifact_store/mlruns

from pipelines.run_training import training_trigger_word_pipeline
from zenml.client import Client
import mlflow

if __name__ == "__main__":
    positive_directory = "data/positives_samples"
    negative_directory = "data/negatives_samples"

    experiment_tracker = Client().active_stack.experiment_tracker
    print(f"Experiment tracking URI: {experiment_tracker.get_tracking_uri()}")

    # Set the experiment name
    mlflow.set_experiment("trigger_detection_word_tracker")

    training_trigger_word_pipeline(
        positive_dir=positive_directory,
        negative_dir=negative_directory,
        epochs=25,
        batch_size=32,
    )

    print("To view the MLflow UI, run:")
    print(f"mlflow ui --backend-store-uri {experiment_tracker.get_tracking_uri()}")

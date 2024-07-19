from zenml import step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml import step
import numpy as np
import librosa
from zenml import pipeline
from zenml.config import DockerSettings
from zenml import step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
import mlflow
from zenml.client import Client
from zenml.logger import get_logger
import numpy as np
import librosa
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from pipelines.run_inference import inference_pipeline

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker
audio_file = "data/new_data/sample-000060.mp3"
pipeline_name = "train_and_deploy_pipeline"
pipeline_step_name = "deploy_model"
model_name = "trigger_word_detection_model"

if __name__ == "__main__":
    audio_file = audio_file
    pipeline_name = pipeline_name
    pipeline_step_name = pipeline_step_name
    model_name = model_name

    inference_pipeline(
        audio_file=audio_file,
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
    )

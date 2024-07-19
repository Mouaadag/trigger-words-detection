from zenml import pipeline

import mlflow
from zenml.client import Client
from zenml.logger import get_logger
import numpy as np
from zenml.config import DockerSettings


from steps.inference_steps.predict_step import predict_step

from steps.inference_steps.load_audio_to_prediction_step import (
    load_audio_to_prediction,
    get_audio_duration,
)
from steps.inference_steps.load_deployed_model_step import (
    load_deployed_model,
)

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker
docker_settings = DockerSettings(required_integrations=["mlflow"])


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(
    audio_file: str,
    pipeline_name: str,
    pipeline_step_name: str,
    model_name: str = "trigger_word_detectionModel",
):
    processed_audio = load_audio_to_prediction(audio_file=audio_file)

    audio_duration = get_audio_duration(audio_file=audio_file)
    service = load_deployed_model(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
    )

    prediction = predict_step(
        service=service,
        mfccs=processed_audio,
        file_name=audio_file,
        audio_duration=audio_duration,
    )
    return prediction

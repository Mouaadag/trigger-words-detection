from zenml import step, log_artifact_metadata, log_step_metadata, ArtifactConfig
from zenml.integrations.mlflow.services import MLFlowDeploymentService
import mlflow
from zenml.client import Client
from zenml.logger import get_logger
import numpy as np
import librosa

import pandas as pd

logger = get_logger(__name__)
from typing_extensions import Annotated, Dict

experiment_tracker = Client().active_stack.experiment_tracker


def estimate_trigger_word_time(mfccs, audio_duration):
    total_frames = mfccs.shape[0]
    frame_duration = audio_duration / total_frames

    start_frame = total_frames // 3
    end_frame = 2 * total_frames // 3

    start_time = start_frame * frame_duration
    end_time = end_frame * frame_duration

    return start_time, end_time


@step(experiment_tracker=experiment_tracker.name)
def predict_step(
    service: MLFlowDeploymentService,
    mfccs: np.ndarray,
    file_name: str,
    audio_duration: float,
    TRIGGER_WORD: str = "boy",
    n_mfcc: int = 13,
    max_pad_len: int = 215,
) -> Annotated[dict, ArtifactConfig(name="prediction")]:
    """Run inference on the deployed model."""
    print("the service running is", service)
    # Run prediction
    try:
        prediction = service.predict(mfccs)
        print("The prediction is: ", prediction)
        # Process prediction
        contains_trigger_word = prediction[0][0] > 0.5
        mfccs_reshaped = mfccs.reshape(n_mfcc, max_pad_len)

        start_ms, end_ms = estimate_trigger_word_time(mfccs_reshaped, audio_duration)
        # Calculate seconds and milliseconds for start_time
        start_seconds = int(start_ms)
        start_milliseconds = int((start_ms - start_seconds) * 1000)

        # Calculate seconds and milliseconds for end_time
        end_seconds = int(end_ms)
        end_milliseconds = int((end_ms - end_seconds) * 1000)

        # Format the result as "2s 34ms"
        start_time_formatted = f"{start_seconds}s {start_milliseconds}ms"
        end_time_formatted = f"{end_seconds}s {end_milliseconds}ms"

        result = {
            "filename": file_name,
            "prediction": float(prediction[0][0]),
            "contains_trigger_word": contains_trigger_word,
        }

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        result = {"filename": file_name, "error": str(e)}

    mlflow.log_params(result)
    print("The file was processed successfully.")
    print("The prediction is: ", result["filename"])
    print("The prediction is: ", result["prediction"])
    print("The prediction is: ", result["contains_trigger_word"])
    # print("The prediction is: ", result.get("trigger_word_time", None))
    if contains_trigger_word:
        result["trigger_word_time"] = (
            "The speaker said the Trigger word :  {0}  between {1}s{2}ms and {3}s{4}ms".format(
                TRIGGER_WORD,
                start_seconds,
                start_milliseconds,
                end_seconds,
                end_milliseconds,
            )
        )
    print("The prediction is: ", result.get("trigger_word_time", None))
    log_artifact_metadata(
        {
            "filename": file_name,
            "contains_trigger_word": str(result["contains_trigger_word"]),
            "prediction": float(result["prediction"]),
            "trigger_word_time": result.get("trigger_word_time", None),
            "start_ms": start_time_formatted,
            "end_ms": end_time_formatted,
        },
    )

    print("Metadata logged")
    return result

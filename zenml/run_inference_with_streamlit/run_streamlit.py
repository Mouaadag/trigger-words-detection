import streamlit as st

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
import mlflow
from zenml.client import Client
from zenml.logger import get_logger
import numpy as np
import librosa
import os

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker
from typing import Dict, Tuple
from typing_extensions import Annotated

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.logger import get_logger

TRIGGER_WORD = "boy"  # Replace with your actual trigger word
import mlflow
import tempfile


@st.cache_resource
def load_deployed_model(
    pipeline_name: str,
    pipeline_step_name: str,
    model_name: str = "trigger_word_detection_model",
) -> MLFlowDeploymentService:
    """Load the deployed model."""
    model_deployer: MLFlowModelDeployer = (
        MLFlowModelDeployer.get_active_model_deployer()
    )

    # Fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
    )
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by step "
            f"'{pipeline_step_name}' in pipeline '{pipeline_name}' with name "
            f"'{model_name}' is currently running."
        )
    # Get the first service (assuming there's only one matching deployment)
    service = existing_services[0]
    print("the service is", service.check_status())
    return service


def process_audio(file_path, sample_rate=16000, n_mfcc=13, max_pad_len=242):
    audio, sr = librosa.load(file_path, sr=sample_rate)

    audio_duration = len(audio) / sr  # Duration in seconds

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = mfccs.T

    if mfccs.shape[0] > max_pad_len:
        mfccs = mfccs[:max_pad_len, :]
    else:
        pad_width = ((0, max_pad_len - mfccs.shape[0]), (0, 0))
        mfccs = np.pad(mfccs, pad_width, mode="constant")

    return mfccs, audio_duration


def estimate_trigger_word_time(mfccs, audio_duration):
    total_frames = mfccs.shape[0]
    frame_duration = audio_duration / total_frames

    start_frame = total_frames // 3
    end_frame = 2 * total_frames // 3

    start_time = start_frame * frame_duration
    end_time = end_frame * frame_duration

    return start_time, end_time


def predict_step(
    service: MLFlowDeploymentService,
    audio_file,
    sample_rate=16000,
    n_mfcc=13,
    max_pad_len=242,
    threshold=0.7,
):
    try:
        mfccs, audio_duration = process_audio(
            audio_file, sample_rate=sample_rate, n_mfcc=n_mfcc, max_pad_len=max_pad_len
        )
        features = np.expand_dims(mfccs, axis=0)
        prediction = service.predict(features)

        contains_trigger_word = prediction[0][0] > threshold

        result = {
            "filename": str(audio_file),
            "prediction": float(prediction[0][0]),
            "contains_trigger_word": contains_trigger_word,
        }

        if contains_trigger_word:
            start_ms, end_ms = estimate_trigger_word_time(mfccs, audio_duration)
            start_seconds = int(start_ms)
            start_milliseconds = int((start_ms - start_seconds) * 1000)
            end_seconds = int(end_ms)
            end_milliseconds = int((end_ms - end_seconds) * 1000)
            result["trigger_word_time"] = (
                "The word {0} was detected between {1}s{2}ms and {3}s{4}ms".format(
                    TRIGGER_WORD,
                    start_seconds,
                    start_milliseconds,
                    end_seconds,
                    end_milliseconds,
                )
            )

        return result

    except Exception as e:
        st.error(f"Error processing audio file: {str(e)}")
        return None


def main(
    pipeline_name: str,
    pipeline_step_name: str,
    model_name: str = "trigger_word_detection_model",
):
    st.title("Audio Trigger Word Detection")

    try:
        deployment_service = load_deployed_model(
            pipeline_name=pipeline_name,
            pipeline_step_name=pipeline_step_name,
            model_name=model_name,
        )
    except Exception as e:
        st.error(f"Error loading MLflow model: {str(e)}")
        return

    uploaded_file = st.file_uploader(
        "Choose an audio file", type=["mp3", "wav", "flac"]
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        if st.button("Detect Trigger Word"):
            result = predict_step(
                service=deployment_service,
                audio_file=tmp_file_path,
            )

            if result:
                st.write(f"Filename: {uploaded_file.name}")
                st.write(f"Prediction: {result['prediction']:.4f}")
                st.write(f"Contains trigger word: {result['contains_trigger_word']}")
                if result["contains_trigger_word"]:
                    st.write(result["trigger_word_time"])
            else:
                st.write("No prediction was made.")

        os.unlink(tmp_file_path)


if __name__ == "__main__":
    pipeline_name = "train_and_deploy_pipeline"
    pipeline_step_name = "deploy_model"
    model_name = "trigger_word_detection_model"
    main(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
    )

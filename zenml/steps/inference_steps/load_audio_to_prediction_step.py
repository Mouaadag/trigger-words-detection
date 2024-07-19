from zenml import step, log_model_metadata, ArtifactConfig, get_step_context

import mlflow
from zenml.client import Client
from zenml.logger import get_logger
import numpy as np
import librosa

# from zenml.materializers.numpy_materializer import NumpyMaterializer

from steps.inference_steps.custom_materializer import NumpyMaterializer

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker
# 1from typing_extensions import Annotated, Dict
from typing import Dict, Tuple
from typing_extensions import Annotated


@step(
    experiment_tracker=experiment_tracker.name, output_materializers=[NumpyMaterializer]
)
def load_audio_to_prediction(
    audio_file: str,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    max_pad_len: int = 215,
) -> Annotated[np.ndarray, ArtifactConfig(name="processed_audio")]:
    """Load audio file from disk."""
    audio, sr = librosa.load(audio_file, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = mfccs.T
    if mfccs.shape[0] > max_pad_len:
        mfccs = mfccs[:max_pad_len, :]
    else:
        pad_width = ((0, max_pad_len - mfccs.shape[0]), (0, 0))
        mfccs = np.pad(mfccs, pad_width, mode="constant")

    # Prepare input for prediction
    mfccs = mfccs.reshape(1, max_pad_len, n_mfcc)

    print("Type of processed audio is", type(mfccs))
    return mfccs


@step(
    experiment_tracker=experiment_tracker.name,
)
def get_audio_duration(audio_file: str) -> float:
    """Get duration of audio file."""
    audio, sr = librosa.load(audio_file, sr=None)
    audio_duration = librosa.get_duration(y=audio, sr=sr)
    return audio_duration

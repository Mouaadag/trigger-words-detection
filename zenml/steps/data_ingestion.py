# steps/data_ingestion.py
from zenml import step, log_artifact_metadata
import librosa
import os
import numpy as np
from typing import Dict
from typing_extensions import Annotated
from zenml.logger import get_logger

logger = get_logger(__name__)


def process_audio(file_path, sample_rate=16000, n_mfcc=13, max_pad_len=215):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=sample_rate)

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Transpose MFCC features to have time as the first dimension
    mfccs = mfccs.T

    # Pad or truncate to max_pad_len
    if mfccs.shape[0] > max_pad_len:
        mfccs = mfccs[:max_pad_len, :]
    else:
        pad_width = ((0, max_pad_len - mfccs.shape[0]), (0, 0))
        mfccs = np.pad(mfccs, pad_width, mode="constant")

    return mfccs


def prepare_dataset(positive_dir, negative_dir):
    X = []
    y = []

    # Process positive samples
    for file in os.listdir(positive_dir):
        if file.endswith(".mp3"):
            file_path = os.path.join(positive_dir, file)
            mfccs = process_audio(file_path)
            X.append(mfccs)
            y.append(1)

    # Process negative samples
    for file in os.listdir(negative_dir):
        if file.endswith(".mp3"):
            file_path = os.path.join(negative_dir, file)
            mfccs = process_audio(file_path)
            X.append(mfccs)
            y.append(0)

    return np.array(X), np.array(y)


@step(enable_cache=True)
def data_ingestion_step(positive_dir: str, negative_dir: str) -> Dict[str, np.ndarray]:
    X, y = prepare_dataset(positive_dir, negative_dir)

    """
    Performs the data ingestion step for a trigger word detection model.

    This function ingests audio data from specified directories containing positive and negative examples, processes the audio files to extract features, and returns the features and labels as numpy arrays.
    
    It also logs metadata about the ingestion process, including the directories used and the shapes of the feature and label arrays.

    Parameters:
    - positive_dir (str): The directory path containing positive samples.
    - negative_dir (str): The directory path containing negative samples.

    Returns:
    - Dict[str, np.ndarray]: A dictionary with keys 'X' and 'y', where 'X' is a numpy array of extracted features from the audio files, and 'y' is a numpy array of labels (1 for positive samples, 0 for negative samples).

    The function is decorated with `@step(enable_cache=True)`, indicating that caching is enabled for this step in the pipeline to avoid reprocessing if the input directories have not changed.
    """
    # log metadata
    log_artifact_metadata(
        {
            "positive_dir": positive_dir,
            "negative_dir": negative_dir,
            "X_shape": str(X.shape),
            "y_shape": str(y.shape),
        }
    )
    return {"X": X, "y": y}

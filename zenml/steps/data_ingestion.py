# # steps/data_ingestion.py
from zenml import step, log_artifact_metadata
import librosa
import os
import numpy as np
from typing import Dict, List
from typing_extensions import Annotated
from zenml.logger import get_logger

logger = get_logger(__name__)


@step(enable_cache=True)
def load_data(positive_dir: str, negative_dir: str) -> Dict[str, List[str]]:
    """
    Loads audio file paths from specified directories for positive and negative examples.

    This function scans the given directories for audio files with an '.mp3' extension and compiles lists of the full paths to these files. It returns a dictionary with two keys, 'positive_files' and 'negative_files', each associated with a list of file paths.

    Parameters:
    - positive_dir (str): The directory path containing positive samples.
    - negative_dir (str): The directory path containing negative samples.

    Returns:
    - Dict[str, List[str]]: A dictionary containing lists of file paths. The keys 'positive_files' and 'negative_files' correspond to the paths of positive and negative audio files, respectively.
    """
    positive_files = [
        os.path.join(positive_dir, file)
        for file in os.listdir(positive_dir)
        if file.endswith(".mp3")
    ]
    negative_files = [
        os.path.join(negative_dir, file)
        for file in os.listdir(negative_dir)
        if file.endswith(".mp3")
    ]

    return {"positive_files": positive_files, "negative_files": negative_files}


@step(enable_cache=True)
def extract_features(
    files: Dict[str, List[str]], sample_rate=16000, n_mfcc=13, max_pad_len=215
) -> Dict[str, np.ndarray]:
    """
    Extracts MFCC features from audio files for trigger word detection.

    This function processes audio files specified in the input dictionary, extracting Mel-frequency cepstral coefficients (MFCCs) for each file.
    It handles both positive and negative examples. The MFCCs are padded or truncated to a fixed length to ensure uniformity.
    The function returns a dictionary with keys 'positive_features' and 'negative_features',
    each containing a numpy array of the extracted features for the respective categories.

    Parameters:
    - files (Dict[str, List[str]]): A dictionary with keys 'positive_files' and 'negative_files', each associated with a list of file paths to process.
    - sample_rate (int, optional): The sampling rate to use when loading audio files. Defaults to 16000.
    - n_mfcc (int, optional): The number of MFCC features to extract. Defaults to 13.
    - max_pad_len (int, optional): The maximum length to pad or truncate the MFCC sequences to. Defaults to 215.

    Returns:
    - Dict[str, np.ndarray]: A dictionary with keys 'positive_features' and 'negative_features',
    each containing a numpy array of the extracted MFCC features for the respective categories.
    """

    def process_audio(file_path):
        audio, sr = librosa.load(file_path, sr=sample_rate)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs = mfccs.T
        if mfccs.shape[0] > max_pad_len:
            mfccs = mfccs[:max_pad_len, :]
        else:
            pad_width = ((0, max_pad_len - mfccs.shape[0]), (0, 0))
            mfccs = np.pad(mfccs, pad_width, mode="constant")
        return mfccs

    positive_features = [process_audio(file) for file in files["positive_files"]]
    negative_features = [process_audio(file) for file in files["negative_files"]]

    return {
        "positive_features": np.array(positive_features),
        "negative_features": np.array(negative_features),
    }


@step(enable_cache=True)
def prepare_dataset(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Prepares the dataset for training by combining positive and negative features.

    This function takes a dictionary containing arrays of positive and negative features,
    concatenates them to form a single feature array (X) and a corresponding label array (y).
    Labels are assigned as 1 for positive features and 0 for negative features. It also logs the shapes of the resulting arrays.

    Parameters:
    - features (Dict[str, np.ndarray]): A dictionary with keys 'positive_features' and 'negative_features', each associated with a numpy array of features.

    Returns:
    - Dict[str, np.ndarray]: A dictionary with keys 'X' and 'y', where 'X' is the combined feature array and 'y' is the array of labels.

    The function logs the shapes of 'X' and 'y' to track the size of the dataset being processed.
    """

    X = np.concatenate(
        (features["positive_features"], features["negative_features"]), axis=0
    )
    y = np.concatenate(
        (
            np.ones(len(features["positive_features"])),
            np.zeros(len(features["negative_features"])),
        ),
        axis=0,
    )

    # Log metadata
    log_artifact_metadata(
        {
            "X_shape": str(X.shape),
            "y_shape": str(y.shape),
        }
    )
    return {"X": X, "y": y}

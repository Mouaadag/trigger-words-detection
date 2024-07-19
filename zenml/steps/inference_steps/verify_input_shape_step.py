from zenml import step
import mlflow
from zenml.client import Client
from zenml.logger import get_logger
import numpy as np
import librosa

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker
from zenml.materializers.numpy_materializer import NumpyMaterializer


@step(
    experiment_tracker=experiment_tracker.name, output_materializers=NumpyMaterializer
)
def verify_input_shape(
    audio_file: str,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    max_pad_len: int = 215,
) -> None:
    """Verify that the input shape matches the model's expectations."""
    audio, sr = librosa.load(audio_file, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = mfccs.T
    if mfccs.shape[0] > max_pad_len:
        mfccs = mfccs[:max_pad_len, :]
    else:
        pad_width = ((0, max_pad_len - mfccs.shape[0]), (0, 0))
        mfccs = np.pad(mfccs, pad_width, mode="constant")

    input_data = mfccs.reshape(1, max_pad_len, n_mfcc)

    assert input_data.shape == (
        1,
        215,
        13,
    ), f"Expected shape (1, 215, 13), got {input_data.shape}"
    print(f"Input shape verified: {input_data.shape}")
    mlflow.log_params({"input_shape": input_data.shape})

from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType
import numpy as np
import os

NUMPY_FILENAME = "data.npy"


class NumpyMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (np.ndarray,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type):
        file_path = os.path.join(self.uri, NUMPY_FILENAME)
        return np.load(file_path, allow_pickle=True)

    def save(self, arr):
        file_path = os.path.join(self.uri, NUMPY_FILENAME)
        np.save(file_path, arr)

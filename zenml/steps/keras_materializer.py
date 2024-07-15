from zenml.materializers.base_materializer import BaseMaterializer
from tensorflow import keras
from typing import Type, Any


# class KerasFunctionalMaterializer(BaseMaterializer):
#     ASSOCIATED_TYPES = (keras.Model,)

#     def handle_input(self, data_type: Type[Any]) -> keras.Model:
#         """Load the model from the artifact store."""
#         return keras.models.load_model(self.uri, compile=False)

#     def handle_return(self, model: keras.Model) -> None:
#         """Save the model to the artifact store."""
#         model.save(self.uri, save_format="keras")


# import os
# from typing import Type, Any
# from zenml.materializers.base_materializer import BaseMaterializer
# from tensorflow import keras


# class KerasFunctionalMaterializer(BaseMaterializer):
#     ASSOCIATED_TYPES = (keras.Model,)

#     def handle_input(self, data_type: Type[Any]) -> keras.Model:
#         """Load the model from the artifact store."""
#         return keras.models.load_model(self.uri, compile=False)


#     def handle_return(self, model: keras.Model) -> None:
#         """Save the model to the artifact store."""
#         model.save(self.uri, save_format="tf")
import os
import tempfile
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Tuple, Type

import tensorflow as tf
from tensorflow.python import keras as tf_keras
from tensorflow.python.keras.utils.layer_utils import count_params

from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.utils import io_utils

if TYPE_CHECKING:
    from zenml.metadata.metadata_types import MetadataType


class KerasMaterializer(BaseMaterializer):
    """Materializer to read/write Keras models."""

    ASSOCIATED_TYPES: ClassVar[Tuple[Type[Any], ...]] = (
        tf.keras.Model,
        tf_keras.Model,
    )
    ASSOCIATED_ARTIFACT_TYPE: ClassVar[ArtifactType] = ArtifactType.MODEL
    MODEL_FILE_NAME = "model.keras"

    def load(self, data_type: Type[Any]) -> tf_keras.Model:
        """Reads and returns a Keras model after copying it to temporary path.

        Args:
            data_type: The type of the data to read.

        Returns:
            A keras.Model model.
        """
        # Create a temporary directory to store the model
        temp_dir = tempfile.TemporaryDirectory()

        # Copy from artifact store to temporary directory
        temp_model_file = os.path.join(temp_dir.name, self.MODEL_FILE_NAME)
        io_utils.copy_dir(self.uri, temp_dir.name)

        # Load the model from the temporary directory
        model = tf.keras.models.load_model(temp_model_file)

        # Cleanup and return
        fileio.rmtree(temp_dir.name)

        return model

    def save(self, model: tf_keras.Model) -> None:
        """Writes a keras model to the artifact store.

        Args:
            model: A keras.Model model.
        """
        # Create a temporary directory to store the model
        temp_dir = tempfile.TemporaryDirectory()
        temp_model_file = os.path.join(temp_dir.name, self.MODEL_FILE_NAME)
        model.save(temp_model_file)
        io_utils.copy_dir(temp_dir.name, self.uri)

        # Remove the temporary directory
        fileio.rmtree(temp_dir.name)

    def extract_metadata(self, model: tf_keras.Model) -> Dict[str, "MetadataType"]:
        """Extract metadata from the given `Model` object.

        Args:
            model: The `Model` object to extract metadata from.

        Returns:
            The extracted metadata as a dictionary.
        """
        return {
            "num_layers": len(model.layers),
            "num_params": count_params(model.weights),
            "num_trainable_params": count_params(model.trainable_weights),
        }

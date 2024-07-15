# run_training.py
from zenml import pipeline
from steps.data_ingestion import data_ingestion_step
from steps.splitting_data import split_dataset
from steps.training import training_step
from steps.evaluation import evaluate_model

# from steps.model_serialization import serialize_model

from zenml import Model, pipeline


model = Model(
    # The name uniquely identifies this model
    # It usually represents the business use case
    name="trigger_word_detection",
    # The version specifies the version
    # If None or an unseen version is specified, it will be created
    # Otherwise, a version will be fetched.
    version=None,
    # Some other properties may be specified
    license="MIT",
    description="Trigger word detection model",
)


@pipeline(enable_cache=False, model=model)
def training_trigger_word_pipeline(
    positive_dir: str,
    negative_dir: str,
    epochs: int,
    batch_size: int,
    test_size: float = 0.01,
    val_size: float = 0.01,
):
    """
    Orchestrates the training pipeline for a trigger word detection model.

    This function defines a pipeline that includes steps for data ingestion, data splitting, model training, and model evaluation.
    It takes directories containing positive and negative examples, the number of epochs for training, batch size, and proportions for test and validation splits as inputs.

    Parameters:
    - positive_dir (str): The directory path containing positive samples.
    - negative_dir (str): The directory path containing negative samples.
    - epochs (int): The number of epochs to train the model.
    - batch_size (int): The size of the batches of data.
    - test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.01.
    - val_size (float, optional): The proportion of the dataset to include in the validation split. Defaults to 0.01.

    Returns:
    - tuple: A tuple containing the trained model and the evaluation results.

    The pipeline is marked with `@pipeline(enable_cache=False, model=model)` decorator, indicating caching is disabled for this pipeline, and it is associated with a specific model configuration.
    """
    # Data ingestion
    data = data_ingestion_step(positive_dir=positive_dir, negative_dir=negative_dir)

    # Data splitting
    split_data = split_dataset(data=data, test_size=test_size, val_size=val_size)

    # Model training
    model = training_step(split_data=split_data, epochs=epochs, batch_size=batch_size)

    # Model serialization
    # serialized_model = serialize_model(model=model)

    # Model evaluation
    evaluation_results = evaluate_model(model=model, split_data=split_data)

    return model, evaluation_results

# run_training.py
from zenml import pipeline
from steps.data_ingestion import load_data, extract_features, prepare_dataset
from steps.splitting_data import split_dataset
from steps.training import training_step
from steps.evaluation import evaluate_model
from zenml import Model, pipeline


model = Model(
    name="trigger_word_detection",
    version=None,
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

    # Load audio file paths from the specified positive and negative directories
    loaded_data = load_data(positive_dir=positive_dir, negative_dir=negative_dir)
    # Extract features (e.g., MFCC) from the loaded audio files
    features = extract_features(files=loaded_data)
    # Prepare the dataset by combining positive and negative features and creating labels
    prepared_data = prepare_dataset(features=features)
    # Split the prepared dataset into training, validation, and test sets
    split_data = split_dataset(
        data=prepared_data, test_size=test_size, val_size=val_size
    )

    # Train the model using the split dataset, specified epochs, and batch size
    model = training_step(split_data=split_data, epochs=epochs, batch_size=batch_size)

    # Evaluate the trained model using the test set from the split data
    evaluation_results = evaluate_model(model=model, split_data=split_data)

    return model, evaluation_results

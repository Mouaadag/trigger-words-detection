from zenml import step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
import mlflow
from zenml.client import Client
from zenml.logger import get_logger


logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def load_deployed_model(
    pipeline_name: str,
    pipeline_step_name: str,
    model_name: str = "trigger_word_detection_model",
) -> MLFlowDeploymentService:
    """Load the deployed model."""
    # Get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

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

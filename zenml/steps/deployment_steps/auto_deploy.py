from typing import Optional
from zenml import ArtifactConfig, get_step_context, step
from zenml.client import Client
from zenml.integrations.mlflow.services.mlflow_deployment import (
    MLFlowDeploymentService,
    MLFlowDeploymentConfig,
)
from zenml.integrations.mlflow.steps.mlflow_deployer import (
    mlflow_model_registry_deployer_step,
)
from zenml.logger import get_logger
from mlflow.tracking import MlflowClient, artifact_utils
import mlflow
from typing import Dict
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from zenml.services import BaseService

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def deploy_if_accurate(evaluation_results: dict) -> bool:
    if evaluation_results["accuracy"] > 0.5:
        return True
    else:
        return False


@step(experiment_tracker=experiment_tracker.name)
def deploy_model(should_deploy: bool) -> Optional[MLFlowDeploymentService]:
    """
    Deploys a model if it meets the accuracy threshold.

    This function checks if the model should be deployed based on the `should_deploy` flag, which is determined by the model's accuracy.
    If the model's accuracy is above a predefined threshold, the function proceeds to deploy the model using ZenML's deployment stack.
    It retrieves the current ZenML client, the model deployer from the active stack, and the experiment tracker.
    The function then fetches the MLflow run ID associated with the current pipeline execution.
    If the model is not deemed accurate enough for deployment, it returns a message indicating so.

    Parameters:
        should_deploy (bool): A flag indicating whether the model meets the accuracy threshold for deployment.

    Returns:
        Optional[MLFlowDeploymentService]: An instance of the MLFlowDeploymentService if the model is deployed; otherwise,
        a message indicating the model is not accurate enough for deployment.

    Note:
        The deployment process is dependent on the ZenML and MLflow configurations of the active stack.
    """
    if not should_deploy:
        return "Model not accurate enough to deploy"

    zenml_client = Client()
    model_deployer = zenml_client.active_stack.model_deployer
    experiment_tracker = zenml_client.active_stack.experiment_tracker

    # Let's get the run id of the current pipeline
    mlflow_run_id = experiment_tracker.get_run_id(
        experiment_name=get_step_context().pipeline.name,
        run_name=get_step_context().pipeline_run.name,
    )

    # Once we have the run id, we can get the model URI using mlflow client
    experiment_tracker.configure_mlflow()
    client = MlflowClient()
    model_name = "trigger_word_detection_model"  # set the model name that was logged
    model_uri = artifact_utils.get_artifact_uri(
        run_id=mlflow_run_id, artifact_path=model_name
    )
    mlflow_deployment_config = MLFlowDeploymentConfig(
        name="mlflow-model-deployment-for-trigger-word-detection",
        description="An example of deploying a model using the MLflow Model Deployer",
        pipeline_name=get_step_context().pipeline.name,
        pipeline_step_name=get_step_context().step_run.name,
        model_uri=model_uri,
        model_name=model_name,
        workers=1,
        mlserver=False,
        timeout=300,
    )

    try:
        service = model_deployer.deploy_model(
            config=mlflow_deployment_config,
            service_type=MLFlowDeploymentService.SERVICE_TYPE,
            replace=True,
            continuous_deployment_mode=True,
        )

        if service:
            print(
                f"Model deployed successfully. Access the endpoint at {service.prediction_url}"
            )
            mlflow.log_param(
                "deployment_pipeline_name", get_step_context().pipeline.name
            )
            # mlflow.log_param("the service deployed is :", service.prediction_url)
        else:
            print("Failed to deploy the model.")
            return None

    except Exception as e:
        print(f"Deployment failed with exception: {e}")
        return None

    mlflow.log_param("deployment_pipeline_name", get_step_context().pipeline.name)
    mlflow.log_param("model_uri", model_uri)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("run_id", mlflow_run_id)
    mlflow.log_param("pipeline_step_name", get_step_context().step_run.name)

    return service

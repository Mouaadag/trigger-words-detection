# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from typing import Optional

from typing_extensions import Annotated

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

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def deploy_model() -> Optional[MLFlowDeploymentService]:
    # Deploy a model using the MLflow Model Deployer
    zenml_client = Client()
    model_deployer = zenml_client.active_stack.model_deployer
    experiment_tracker = zenml_client.active_stack.experiment_tracker
    # Let's get the run id of the current pipeline
    mlflow_run_id = experiment_tracker.get_run_id(
        experiment_name=get_step_context().pipeline.name,
        run_name=get_step_context().pipeline_run.name,
    )
    # mlflow_run_id = "07a4ab368f0647d18a0803367a7525db"
    # Once we have the run id, we can get the model URI using mlflow client
    experiment_tracker.configure_mlflow()
    # client = MlflowClient()
    model_name = "trigger_word_detectionModel"  # set the model name that was logged
    model_uri = artifact_utils.get_artifact_uri(
        run_id=mlflow_run_id, artifact_path=model_name
    )
    mlflow_deployment_config = MLFlowDeploymentConfig(
        name="mlflow-model-deployment-example",
        description="An example of deploying a model using the MLflow Model Deployer",
        pipeline_name=get_step_context().pipeline.name,
        pipeline_step_name=get_step_context().step_run.name,
        model_uri=model_uri,
        model_name=model_name,
        workers=1,
        mlserver=False,
        timeout=300,
    )
    # service = model_deployer.deploy_model(mlflow_deployment_config)
    service = model_deployer.deploy_model(
        config=mlflow_deployment_config,
        service_type=MLFlowDeploymentService.SERVICE_TYPE,
    )
    mlflow.log_param("deployment_pipeline_name", get_step_context().pipeline.name)

    return service
    ### YOUR CODE ENDS HERE ###

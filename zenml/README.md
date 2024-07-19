To create a ZenML experiment, deploy an artifact store, orchestrator, and register a stack, follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the root directory of your ZenML project using the `cd` command.
3. Create a new experiment by running the following command:

   ```
   zenml experiment-tracker register <experiment_name> --flavor=mlflow\
   ```

   Replace `<experiment_name>` with the desired name for your experiment.

4. Create a model-deployer an artifact store by executing the following command:

   ```
   zenml model-deployer register <model-deployer_name> --flavor=mlflow\
   ```

   Replace `<model-deployer_name>` with the desired name for your experiment.

5. Create an artifact store

   ```
   zenml artifact-store register <artifact_store_name> --flavor=local --path=<artifact_store_type>
   ```

   Replace `<artifact_store_name>` with the name of your artifact store and `<artifact_store_type>` with the desired type (e.g., `local path`, `gcs`, `s3`).

6. Set up an orchestrator by running the following command:

   ```
    zenml orchestrator register <orchestrator_name> --flavor=<orchestrator_type>\
   ```

   Replace `<orchestrator_name>` with the name of your orchestrator and `<orchestrator_type>` with the desired type (e.g., `airflow`, `kubeflow`, `default`).

7. Install zenml integrations for mlflow so we can track experiments.

```
 zenml integration install mlflow -y
```

8. Once the MLflow integration is installed, you can register an MLflow model registry component in your stack:

```
 zenml model-registry register mlflow_model_registry --flavor=mlflow

```

8. Register a stack by executing the following command:
   ```
    zenml stack register <stack_name> \\
    -e <experiment_name> \\
    -o default_orchestrator \\
    -a <artifact_store_name> \\
    -d <model-deployer_name> \\
    -r <model-registry>
   ```
   Replace `<stack_name>` with the name of your stack.

Remember to adjust the commands and options based on your specific project requirements and environment setup.

In this project I created pipeline to train and deploy a model using Mlflow,

To execute the `run_training_pipeline` in the `zenml` project, follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the root directory of the `zenml` project using the `cd` command.
3. Run the following command to activate the virtual environment (if applicable):
   ```
   source zenml-venv/bin/activate
   ```
4. Run the `run_train_and_deploy_pipeline.py` script by executing the following command:
   ```
   python run_train_and_deploy_pipeline.py
   ```
5. Wait for the training pipeline to complete. You should see the progress and any output or logs generated during the process.
6. Once the training pipeline finishes, you can analyze the results or perform any further actions as required.

7. To run inference with `streamlit` you can execute the command :
   ```
   streamlit run run_inference_with_streamlit/run_streamlit.py
   ```

Remember to adjust the commands based on your specific project structure and environment setup.

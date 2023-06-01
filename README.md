# End-to-End CV MLOps Project

![CI & Tests](https://github.com/data42lana/end-to-end-cv-mlops-project/actions/workflows/ci-tests.yml/badge.svg) ![python: 3.9](https://img.shields.io/badge/%20python-3.9-blue) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) [![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit) ![project status: not updated](https://img.shields.io/badge/project%20status-not%20updated-red)

> ### The source code and some results of local end-to-end implementation of MLOps in practice for an object detection web application project, including the source code of the application itself.

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#app-demo">App Demo</a> •
  <a href="#how-to-test">How To Test</a>
</p>

## About
The goal of creating this project was to learn how to implement end-to-end MLOps in practice. As a result, the machine learning pipeline was built that fine-tunes a pre-trained model and the web application for detecting house sparrows in a photo that interacts with the model via an API. This repository contains the source code of the pipeline, including the source code of the web app and API, and some of its run results, which are needed to reproduce the pipeline and demonstrate its work. The following diagram shows more clearly how MLOps is implemented in the project:

![Project MLOps Diagram](./docs/project-mlops-diagram.svg)

Its extended version can be viewed in `docs/project-mlops-diagram-extended.svg`.

> More information about the dataset and model used can be found in `docs/dataset-card.md` and `docs/model-card.md`, respectively.

## Installation
The source code was developed on Windows. All the Python modules in the repository, except those in the `deployment` folder, were also run and tested on Linux (Ubuntu) (in *Google Colab* and using *GitHub Actions*, respectively).

First, clone this repo and go to its root directory. Then, create a virtual environment with Python 3.9 (not tested on other versions) and activate it. After that, install either all the dependencies of the project by running from the command line:
```bash
$ python -m pip install -r requirements/dev-requirements.txt
```
or only those needed for a specific MLOps task (see table below).

> **Note** The repo contains two types of ML pipelines: DVC pipelines (including `pipelines/dvc.yaml` and `pipelines/new_data_pipeline/dvc.yaml`) and a Metaflow workflow (`pipelines/mlworkflow.py`) (**not available for Windows!**). If you plan to use the second one, then uncomment the appropriate tool in the `pipe-requirements.txt` file to install it.

<details>
  <summary><b>Show requirements table</b></summary>
    <table>
      <tr>
        <th>MLOps Component/Task</th>
        <th>Requirements</th>
        <th>Used Files & Folders</th>
        <th>Output Files & Folders</th>
      </tr>
      <tr>
        <td>EDA</td>
        <td>eda-requirements.txt</td>
        <td>notebooks/EDA.ipynb, data/raw</td>
        <td>notebooks/EDA.ipynb (outputs), data/prepared (optional)</td>
      </tr>
      <tr>
        <td>Data Checking</td>
        <td>data-check-requirements.txt</td>
        <td>data_checks, great_expectations, data, configs/params.yaml</td>
        <td>data_checks/data_check_results, great_expectations/uncommitted, pipe.log</td>
      </tr>
      <tr>
        <td>Model Training</td>
        <td>train-requirements.txt</td>
        <td>src, data, configs</td>
        <td>hyper_opt, configs/best_params.yaml, mlruns, models, outputs, reports/model_report.md, pipe.log</td>
      </tr>
      <tr>
        <td>Pipeline/Workflow</td>
        <td>pipe-requirements.txt</td>
        <td>pipelines, data_checks, great_expectations, src, data, configs</td>
        <td>.dvc, pipelines (/dvc.lock & /dvc_dag.md) or .metaflow, data_checks/data_check_results, great_expectations, hyper_opt, configs/best_params.yaml, mlruns, models, outputs, reports/model_report.md, pipe.log</td>
      </tr>
      <tr>
        <td>Model Deployment / API & App</td>
        <td>deployment-requirements.txt</td>
        <td>deployment (except /demo), src/train/train_inference_fns.py, src/utils.py, mlruns, configs/params.yaml, .streamlit</td>
        <td>monitoring/current_deployed_model.yaml, monitoring/data</td>
      </tr>
      <tr>
        <td>Model Monitoring</td>
        <td>monitoring-requirements.txt</td>
        <td>monitoring, data, configs/params.yaml</td>
        <td>monitoring/deployed_model_check_results, reports/deployed_model_performance_report.html, mon.log</td>
      </tr>
      <tr>
        <td>Continuous Integration (CI)</td>
        <td>ci-requirements.txt</td>
        <td>.github, tests (except /webapi), pytest.ini, data_checks, src</td>
        <td>-</td>
      </tr>
      <tr>
        <td>Web App Demo</td>
        <td>requirements.txt (in deployment/demo)</td>
        <td>deployment/demo</td>
        <td>-</td>
      </tr>
    </table>
</details>

If you used `dev-requirements.txt`, run `pre-commit install` to install git hook scripts from the `.pre-commit-config.yaml` file (including for the DVC project in this repo). If you want to use your own Great Expectations/DVC projects, ensure that they are initialized in the root directory of the repo, or do it by running the `great_expectations init` / `dvc init` commands. For details, refer to the documentation for the respective tools.

The dataset to reproduce the ML pipeline of this project can be found [here](https://kaggle.com/datasets/c09594815010b1f11dd842fc1dc1128b28f470b2034412dd3a4444ac979c25b3). To run the pipeline on your own data, they must be organized as described in `docs/dataset-card.md`. If necessary, configure items of the Great Expectations project according to the new data. See samples of the used data in the `tests/data_samples` folder.

## Configuration
The project is configured to run on a local machine, although this can be changed if necessary. The main settings of the MLOps for this project are held in the `configs/params.yaml` file.

> **Note** To use remote storages or advanced features, some installed Python packages, DVC and MLflow for example, require additional dependencies to be installed. See their documentation.

## How To Use
Below are the CLI commands for MLOps components, which are executed manually in this implementation. Their order matters because the following commands depend on the results of the previous ones. Other components are already included in the pipelines/workflow, such as data verification/validation, hyperparameter optimization, and model stage transition to production, or are triggered when code is pushed to GitHub, such as tests.

1. Run either the pipeline or the workflow (both contain similar stages/steps) to train (fine-tune) a object detection model:
-
    ```bash
    # (Optional) Generate a Python script for the 'new_data_expectation_check' step
    $ great_expectations checkpoint script new_image_info_and_bbox_ckpt.yml

    # Generate a Python script for the 'raw_data_expectation_check' step
    $ great_expectations checkpoint script image_info_and_bbox_ckpt.yml

    # Run the model training workflow
    $ python pipelines/mlworkflow.py run
    ```
    or with the `--production` flag if the trained model will be used in production, regardless of its performance.
    > **Warning** The workflow is created using Metaflow, which is not available on Windows.
-
    ```bash
    # (Optional) Reproduce the new data check pipeline
    $ dvc repro pipelines/new_data_pipeline/dvc.yaml

    # Reproduce the model training pipeline without including new data checks
    $ dvc repro pipelines/dvc.yaml
    ```
    or use the `--all-pipelines` flag to reproduce all the pipelines for all the `dvc.yaml` files present in the repo. DAGs of the pipelines can be viewed in the `pipelines/dvc_dag.md` file.

2. (Optional) View a history of model training runs:
-
    ```bash
    $ mlflow ui --backend-store-uri sqlite:///mlruns/mlruns.db
    ```
    > **Note** Change the value for `--backend-store-uri` to match the tracking server URI set for MLflow.

3. Use a deployed model via the API in the web app to get its performance data:
-
    ```bash
    # Run the API on uvicorn server
    $ python deployment/api.py

    # Run the web app on Streamlit server
    $ streamlit run deployment/app.py
    ```

4. Monitor the performance of the deployed model:
-
    ```bash
    $ python monitoring/monitor_deployed_model.py
    ```
As a result of executing the commands, the project directory will have the structure similar to that presented in the `docs/project-directory-structure.md` file.

Notebooks in this repo, except `EDA.ipynb`, contain trial runs of the data checks and initial experiments to build the model training pipeline.

## App Demo
You can try out the web application from this project on [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/data42lana/how-many-house-sparrows-demo).

If the demo is not available or not displayed, then it can be seen as a static image in `docs/app-image.pdf`.

## How To Test
If pytest and its required plugins are not installed, run from the command line:
```bash
$ python -m pip install -r requirements/test-requirements.txt
```
Test configurations are held in the `pytest.ini` file.

```bash
# Run all the tests in the repo except API ones
$ pytest --ignore=tests/webapi/ tests/

# Run uvicorn server and the API
$ python deployment/api.py

# Run API tests
$ pytest tests/webapi/
```
> **Note** Some of the tests take a long time, they are marked as "slow".
> ```bash
> # Skip slow tests
> $ pytest -m "not slow" tests/
> ```

> **Warning** Sometimes the `integration` tests fail. This is due to the stochastic nature of machine learning algorithms. Try running the test again.

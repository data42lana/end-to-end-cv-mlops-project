"""This module creates a workflow for the machine learning project."""
# Note: Metaflow does not run on Windows!

import os
# import sys
from pathlib import Path
# sys.path.append(str(Path.cwd()))

from metaflow import (Flow, FlowSpec, Parameter, card, catch, current, project, retry,
                      step, timeout)
from metaflow.cards import Image, Markdown

from src.utils import get_param_config_yaml

PROJECT_PATH = Path.cwd()
MLCONFIG = get_param_config_yaml(PROJECT_PATH)
MLTRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI',
                                MLCONFIG['mlflow_tracking_conf']['mltracking_uri'])

# Additional options
SAVE_EDA_PLOTS_TO_FILES = True
SAVE_METRIC_PLOTS_TO_FILES = True
COMPARE_WITH_PRODUCTION_MODEL = True


@project(name='end_to_end_cv_mlops_project')
class MLWorkFlow(FlowSpec):
    """A flow containing steps of working with data, optimizing
    hyperparameters, training a model, and preparing it for production.
    """

    mltracking_uri = Parameter('mltracking_uri', default=MLTRACKING_URI)

    @retry(times=1, minutes_between_retries=5)
    @step
    def start(self):
        """Check if raw and new data is available."""
        import numpy as np
        data_is_available = []
        for data_path in ['image_data_paths', 'new_image_data_paths']:
            data_path_exists = np.all(
                [PROJECT_PATH.joinpath(MLCONFIG[data_path][dpath]).exists()
                 for dpath in MLCONFIG[data_path]
                 if dpath not in ['train_csv_file', 'test_csv_file']])
            data_is_available.append(data_path_exists)
        self.raw_data_is_available, self.new_data_is_available = data_is_available
        if not self.raw_data_is_available:
            raise ValueError("Raw data is not available!")
        self.next(self.new_data_expectation_check)

    @step
    def new_data_expectation_check(self):
        """Check new data against predefined expectations."""
        from great_expectations.checkpoint.types.checkpoint_result import (
            CheckpointResult)
        from great_expectations.data_context import DataContext
        if self.new_data_is_available:
            data_context = DataContext(
                context_root_dir=str(PROJECT_PATH.joinpath('great_expectations')))
            result: CheckpointResult = data_context.run_checkpoint(
                checkpoint_name='new_image_info_and_bbox_ckpt',
                batch_request=None,
                run_name=None)
            if not result['success']:
                raise ValueError("New Data Expectation Check is failed!")
        else:
            print("New Data Expectation Check is skipped: New data is not available!")
        self.next(self.new_data_integrity_check)

    @step
    def new_data_integrity_check(self):
        """Check new data for integrity."""
        from data_checks.check_img_info_and_bbox_csv_file_integrity import (
            main as check_csv_integrity)
        if self.new_data_is_available:
            check_passed = check_csv_integrity(PROJECT_PATH, MLCONFIG, 'new',
                                               PROJECT_PATH / 'data_checks')
            if not check_passed:
                raise ValueError("New Data Integrity Check is failed!")
        else:
            print("New Data Integrity Check is skipped: New data is not available!")
        self.next(self.new_data_similarity_check)

    @step
    def new_data_similarity_check(self):
        """Check new and raw datasets for similarity."""
        from data_checks.check_bbox_duplicates_and_two_dataset_similarity import (
            main as check_two_dataset_similarity)
        if self.new_data_is_available:
            check_passed = check_two_dataset_similarity(PROJECT_PATH, MLCONFIG, 'new',
                                                        PROJECT_PATH / 'data_checks')
            if not check_passed:
                raise ValueError("New Data Similarity Check is failed!")
        else:
            print("New Data Similarity Check is skipped: New data is not available!")
        self.next(self.adding_new_data_to_raw)

    @step
    def adding_new_data_to_raw(self):
        """Add new images to raw ones and update raw csv files."""
        from src.data.update_raw_data import main as update_raw_data
        if self.new_data_is_available:
            update_raw_data(PROJECT_PATH, MLCONFIG)
        else:
            print("Adding New Data to Raw is skipped: New data is not available!")
        self.next(self.raw_data_expectation_check)

    @step
    def raw_data_expectation_check(self):
        """Check raw data against predefined expectations."""
        from great_expectations.checkpoint.types.checkpoint_result import (
            CheckpointResult)
        from great_expectations.data_context import DataContext
        data_context = DataContext(
            context_root_dir=str(PROJECT_PATH.joinpath('great_expectations')))
        result: CheckpointResult = data_context.run_checkpoint(
            checkpoint_name='image_info_and_bbox_ckpt',
            batch_request=None,
            run_name=None)
        if not result['success']:
            raise ValueError("Raw Data Expectation Check is failed!")
        self.next(self.raw_data_bbox_duplication_check)

    @step
    def raw_data_bbox_duplication_check(self):
        """Check raw data for bbox duplicates."""
        from data_checks.check_bbox_duplicates_and_two_dataset_similarity import (
            main as check_bbox_csv_duplication)
        check_passed = check_bbox_csv_duplication(PROJECT_PATH, MLCONFIG, 'raw',
                                                  PROJECT_PATH / 'data_checks')
        if not check_passed:
            raise ValueError("Raw Data Duplication Check is failed!")
        self.next(self.raw_data_integrity_check)

    @step
    def raw_data_integrity_check(self):
        """Check raw data for integrity."""
        from data_checks.check_img_info_and_bbox_csv_file_integrity import (
            main as check_csv_integrity)
        check_passed = check_csv_integrity(PROJECT_PATH, MLCONFIG, 'raw',
                                           PROJECT_PATH / 'data_checks')
        if not check_passed:
            raise ValueError("Raw Data Integrity Check is failed!")
        self.next(self.train_test_data_split)

    @card(type='blank')
    @step
    def train_test_data_split(self):
        """Split raw data into training and test sets and generate a EDA report
        of the training data using card components.
        """
        from src.data.prepare_data import main as split_data_into_train_test
        self.eda_plots = split_data_into_train_test(PROJECT_PATH, MLCONFIG,
                                                    save_eda_plots=SAVE_EDA_PLOTS_TO_FILES)
        # Set a EDA report title
        current.card.append(Markdown("# EDA Report"))
        # Add training eda plots
        current.card.append(Markdown("## Train Dataset Plot(s):"))
        for plot in self.eda_plots:
            current.card.append(Image.from_matplotlib(plot))
        self.next(self.train_test_similarity_check)

    @step
    def train_test_similarity_check(self):
        """Check training and test datasets for similarity."""
        from data_checks.check_bbox_duplicates_and_two_dataset_similarity import (
            main as check_two_dataset_similarity)
        check_passed = check_two_dataset_similarity(PROJECT_PATH, MLCONFIG, 'prepared',
                                                    PROJECT_PATH / 'data_checks')
        if not check_passed:
            raise ValueError("Train Test Similarity Check is failed!")
        self.next(self.hyperparam_optimization)

    @retry(times=0)
    @step
    def hyperparam_optimization(self):
        """Find the best hyperparameters for model training."""
        from src.train.optimize_hyperparams import main as optimize_hyperparams
        optimize_hyperparams(PROJECT_PATH, MLCONFIG)
        self.next(self.model_fine_tuning)

    @retry(times=0)
    @timeout(hours=1)
    @step
    def model_fine_tuning(self):
        """Fine-tune a model on a specific dataset."""
        import mlflow
        from src.train.fine_tune_model import main as fine_tune_model
        mlflow.set_tracking_uri(self.mltracking_uri)
        fine_tune_model(PROJECT_PATH, MLCONFIG)
        self.next(self.model_performance_on_test_data)

    @step
    def model_performance_on_test_data(self):
        """Get model performance on a test dataset."""
        import mlflow
        from src.train.model_test_performance import main as run_model_test_performance
        mlflow.set_tracking_uri(self.mltracking_uri)
        self.test_res = run_model_test_performance(
            PROJECT_PATH, MLCONFIG, get_random_prediction_image=True,
            compare_with_production_model=COMPARE_WITH_PRODUCTION_MODEL)
        self.test_score = [self.test_res['test_score_name'],
                           round(self.test_res['test_score_value'], 2)]
        self.mlflow_model_uri = self.test_res['model_uri']
        self.next(self.model_stage_update)

    @retry(times=0)
    @step
    def model_stage_update(self):
        """Update model version stages and tags to 'production' or 'archived'."""
        import mlflow
        from src.model.update_model_stages import main as update_model_version_stages

        # Get a production test score
        prod_runs = list(Flow(current.flow_name).runs('production'))
        if prod_runs:
            prod_test_score = max([prod_run[current.step_name].task.data.test_score[1]
                                   for prod_run in prod_runs])
        else:
            prod_test_score = 0

        # Compare the current and production test scores
        if (self.test_score[1] > prod_test_score) or current.is_production:
            mlflow.set_tracking_uri(self.mltracking_uri)
            res = update_model_version_stages(PROJECT_PATH, MLCONFIG,
                                              SAVE_METRIC_PLOTS_TO_FILES)
            self.prod_run_id_in_mlflow, self.prod_model_id_in_mlflow, self.prod_metric_plots = res  # noqa: B950

            # Reassign 'production' tags
            for prod_run in prod_runs:
                prod_run.replace_tag('production', 'archived')
            Flow(current.flow_name)[current.run_id].add_tag('production')
        self.next(self.end)

    @catch
    @card(type='blank')
    @step
    def end(self):
        """Generate a report using card components."""
        import mlflow
        from src.utils import (get_current_stage_of_registered_model_version,
                               get_number_of_csv_rows)

        # Get training and test dataset sizes
        data_paths = MLCONFIG['image_data_paths']
        self.train_dataset_size, self.test_dataset_size = [
            get_number_of_csv_rows(data_paths[csv_file])
            for csv_file in ['train_csv_file', 'test_csv_file']]

        # Get the current stage of the model version
        mlflow.set_tracking_uri(self.mltracking_uri)
        client = mlflow.MlflowClient()
        _, model_name, model_version = self.test_res['model_uri'].split('/')
        self.current_model_stage = get_current_stage_of_registered_model_version(
            client, model_name, int(model_version))

        # Set a report title
        current.card.append(Markdown("# Model Training WorkFlow Result Report"))

        # Add model information
        current.card.append(Markdown("## Model Information"))
        model_info = {"Registered Name": f"**{model_name}**",
                      "Version": f"**{model_version}**",
                      "Stage": f"**{self.current_model_stage}**"}
        for sname, sval in model_info.items():
            current.card.append(Markdown(f"*{sname}:* **{sval}**"))

        # Add model performance on test data
        current.card.append(Markdown("### Performance on Test Data"))
        model_test_perform = {
            self.test_score[0].capitalize(): "**{}**".format(self.test_score[1]),
            "Test Dataset Size": self.test_dataset_size}
        for sname, sval in model_test_perform.items():
            current.card.append(Markdown(f"*{sname}:* {sval}"))

        if 'test_img_info' in self.test_res:
            # Add an image with test prediction result
            current.card.append(Markdown("*Example:*"))
            current.card.append(Markdown(
                "The Number of House Sparrows on the Image: {}".format(
                    self.test_res['test_predict_number'])))
            current.card.append(Image.from_matplotlib(self.test_res['test_predict_img']))

            # Create a link to the image
            photo_author = self.test_res['test_img_info']['Author']
            photo_source = self.test_res['test_img_info']['Source']
            photo_source_link = 'https://{0}.com'.format(photo_source.lower())
            photo_number = self.test_res['test_img_info']['Name'].split('_', maxsplit=1)[0]
            photo_license = self.test_res['test_img_info']['License'].split(')')[0].split('(')[1]  # noqa: B950
            current.card.append(Markdown(
                "Photo by {0} on [{1}]({2}). No {3}. License: {4} "
                "*The photo modified: boxes and scores drawn*.".format(
                    photo_author, photo_source, photo_source_link,
                    photo_number, photo_license)))

        # Expand the report for a production model
        if 'production' in Flow(current.flow_name)[current.run_id].user_tags:
            self.load_params = MLCONFIG['object_detection_model']['load_parameters']
            self.train_backbone_layers = self.load_params.get('trainable_backbone_layers', 3)
            self.max_number_detections = self.load_params.get('box_detections_per_img', 100)

            # Add model training details
            current.card.append(Markdown("## Model Training Details"))
            model_train_details = {
                "MLflow Run Id": self.prod_run_id_in_mlflow,
                "Training Dataset Size": self.train_dataset_size,
                "Number of Trained Backbone Layers": self.train_backbone_layers,
                "Maximum Number of House Sparrows (that can be detected in one image)": self.max_number_detections}  # noqa: B950
            for sname, sval in model_train_details.items():
                current.card.append(Markdown(f"*{sname}:* {sval}"))

            # Add training metric history plots
            current.card.append(Markdown("### Metric History Plot(s):"))
            for plot in self.prod_metric_plots:
                current.card.append(Image.from_matplotlib(plot))


if __name__ == '__main__':
    if 'USERNAME' not in os.environ:
        os.environ['USERNAME'] = 'user1'
    MLWorkFlow()

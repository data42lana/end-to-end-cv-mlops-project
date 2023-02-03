"""This module creates a workflow for this ml project."""
from pathlib import Path

from metaflow import (Flow, FlowSpec, Parameter, card, current, project, retry, step,
                      timeout)
from metaflow.cards import Image, Markdown

from src.utils import get_config_yml

MLCONFIG = get_config_yml()
PROJECT_PATH = Path.cwd()
MLTRACKING_URI = 'sqlite:///mlruns/mlruns.db'


@project(name='object_detection_with_mlops_project')
class MLWorkFlow(FlowSpec):
    """A flow containing steps of working with data, optimizing
    hyperparameters, training a model, and preparing it for production.
    """

    mltracking_uri = Parameter('mltracking_uri', default=MLTRACKING_URI)

    @retry(times=1)
    @step
    def start(self):
        """Check if raw data is available."""
        import numpy as np
        self.raw_data_is_available = np.all([PROJECT_PATH.joinpath(data).exists()
                                             for data in MLCONFIG['image_data_paths']])
        if not self.raw_data_is_available:
            raise FileNotFoundError("The raw data is not available!")
        self.next(self.new_data_availability_check)

    @retry(times=1)
    @step
    def new_data_availability_check(self):
        """Check if new data is available."""
        import numpy as np
        self.new_data_is_available = np.all([PROJECT_PATH.joinpath(data).exists()
                                             for data in MLCONFIG['new_image_data_paths']])
        if self.new_data_is_available:
            self.next(self.new_data_expectation_check)
        else:
            self.next(self.data_expectation_check)

    @step
    def new_data_expectation_check(self):
        """Check new data against defined expectations."""
        from great_expectations.checkpoint.types.checkpoint_result import (
            CheckpointResult)
        from great_expectations.data_context import DataContext
        data_context = DataContext(
            context_root_dir=str(PROJECT_PATH.joinpath('great_expectations')))
        result: CheckpointResult = data_context.run_checkpoint(
            checkpoint_name='new_image_info_and_bbox_ckpt',
            batch_request=None,
            run_name=None)
        if result['success']:
            self.next(self.new_data_duplication_check)
        else:
            self.next(self.end)

    @step
    def new_data_duplication_check(self):
        """Check new data for duplicates."""
        from data_checks.check_duplicates_and_two_dataset_similarity import (  # noqa: B950
            main as check_csv_duplication)
        check_passed = check_csv_duplication(PROJECT_PATH, MLCONFIG, 'new',
                                             PROJECT_PATH / 'data_checks')
        if check_passed:
            self.next(self.new_data_integrity_check)
        else:
            self.next(self.end)

    @step
    def new_data_integrity_check(self):
        """Check new data for integrity."""
        from data_checks.check_img_info_and_bbox_csv_file_integrity import (  # noqa: B950
            main as check_csv_integrity)
        check_passed = check_csv_integrity(PROJECT_PATH, MLCONFIG, 'new',
                                           PROJECT_PATH / 'data_checks')
        if check_passed:
            self.next(self.new_data_similarity_check)
        else:
            self.next(self.end)

    @step
    def new_data_similarity_check(self):
        """Check new and raw datasets for similarity."""
        from data_checks.check_duplicates_and_two_dataset_similarity import (  # noqa: B950
            main as check_two_dataset_similarity)
        check_passed = check_two_dataset_similarity(PROJECT_PATH, MLCONFIG, 'new',
                                                    PROJECT_PATH / 'data_checks')
        if check_passed:
            self.next(self.adding_new_data_to_raw)
        else:
            self.next(self.end)

    @step
    def adding_new_data_to_raw(self):
        """Add new images to raw ones and update raw csv files."""
        from src.data.update_raw_data import main as update_raw_data
        update_raw_data(PROJECT_PATH, MLCONFIG)
        self.next(self.data_expectation_check)

    @step
    def data_expectation_check(self):
        """Check raw data against defined expectations."""
        from great_expectations.checkpoint.types.checkpoint_result import (
            CheckpointResult)
        from great_expectations.data_context import DataContext
        data_context = DataContext(
            context_root_dir=str(PROJECT_PATH.joinpath('great_expectations')))
        result: CheckpointResult = data_context.run_checkpoint(
            checkpoint_name='image_info_and_bbox_ckpt',
            batch_request=None,
            run_name=None)
        if result['success']:
            self.next(self.data_duplication_check)
        else:
            self.next(self.end)

    @step
    def data_duplication_check(self):
        """Check raw data for duplicates."""
        from data_checks.check_duplicates_and_two_dataset_similarity import (  # noqa: B950
            main as check_csv_duplication)
        check_passed = check_csv_duplication(PROJECT_PATH, MLCONFIG, 'raw',
                                             PROJECT_PATH / 'data_checks')
        if check_passed:
            self.next(self.data_integrity_check)
        else:
            self.next(self.end)

    @step
    def data_integrity_check(self):
        """Check raw data for integrity."""
        from data_checks.check_img_info_and_bbox_csv_file_integrity import (  # noqa: B950
            main as check_csv_integrity)
        check_passed = check_csv_integrity(PROJECT_PATH, MLCONFIG, 'raw',
                                           PROJECT_PATH / 'data_checks')
        if check_passed:
            self.next(self.train_test_data_split)

    @step
    def train_test_data_split(self):
        """Split raw data into traininig and test sets."""
        from src.data.prepare_data import main as split_data_into_train_test
        split_data_into_train_test(PROJECT_PATH, MLCONFIG)
        self.next(self.data_similarity_check)

    @step
    def data_similarity_check(self):
        """Check training and test datasets for similarity."""
        from data_checks.check_duplicates_and_two_dataset_similarity import (  # noqa: B950
            main as check_two_dataset_similarity)
        check_passed = check_two_dataset_similarity(PROJECT_PATH, MLCONFIG, 'prepared',
                                                    PROJECT_PATH / 'data_checks')
        if check_passed:
            self.next(self.hyperparam_optimization)

    @step
    def hyperparam_optimization(self):
        """Find the best hyperparameters for model training."""
        from src.train.optimize_hyperparams import main as optimize_hyperparams
        optimize_hyperparams(PROJECT_PATH, MLCONFIG)
        self.next(self.train)

    @timeout(minutes=45)
    @step
    def model_fine_tuning(self):
        """Fine-tune a model on a specific dataset."""
        import mlflow

        from src.train.fine_tune_model import main as fine_tune_model
        mlflow.set_tracking_uri(self.mltracking_uri)
        fine_tune_model(PROJECT_PATH, MLCONFIG)
        self.next(self.model_inference_on_test_data)

    @step
    def model_inference_on_test_data(self):
        """Run a model inference on a test dataset."""
        # Get a current test score
        import mlflow

        from src.model.model_test_inference import main as run_model_test_inference
        mlflow.set_tracking_uri(self.mltracking_uri)
        self.test_res = run_model_test_inference(PROJECT_PATH, MLCONFIG,
                                                 get_random_prediction=True)
        self.test_score = [self.test_res['test_score_name'],
                           self.test_res['test_score_value']]

        # Get a production test score
        prod_runs = list(Flow(current.flow_name).runs('production'))
        if prod_runs:
            if len(prod_runs) == 1:
                prod_test_score = prod_runs[0][current.step_name].task.data.test_score
            else:
                prod_test_score = max([prod_run[current.step_name].task.data.test_score
                                       for prod_run in prod_runs])
        else:
            prod_test_score = 0

        # Compare the current and production test scores
        if (self.test_score[1] > prod_test_score) or current.is_production:
            self.next(self.model_stage_update)
        else:
            self.next(self.end)

    @step
    def model_stage_update(self):
        """Update model version stages and tags to 'production' or 'archived.'"""
        import mlflow

        from src.model.update_model_stages import main as update_model_version_stages
        mlflow.set_tracking_uri(self.mltracking_uri)
        self.prod_run_id_in_mlflow, self.prod_metric_plots = update_model_version_stages(
            PROJECT_PATH, MLCONFIG)

        # Reassign 'production' tags
        prod_runs = Flow(current.flow_name).runs('production')
        for prod_run in prod_runs:
            prod_run.replace_tag('production', 'archived')
        Flow(current.flow_name)[current.run_id].add_tag('production')
        self.next(self.production_report_generation)

    @card(type='blank')
    @step
    def production_report_generation(self):
        """Generate a report for a production model using card components."""
        # Set a report title
        current.card.append(Markdown("# Model Performance Report"))

        # Add a current test score
        current.card.append(Markdown("## Test {0} score: {1:0.2f}".format(
            self.test_score[0], self.test_score[1])))

        # Add training metric plots
        current.card.append(Markdown("## Metric History Plots:"))
        for plot in self.prod_metric_plots:
            current.card.append(Image.from_matplotlib(plot))

        # Add a test prediction result
        current.card.append(Markdown("## Model Test Prediction:"))
        current.card.append(Markdown("The Number of House Sparrows on the Image: {}".format(
            self.test_res['test_predict_number'])))
        current.card.append(Image.from_matplotlib(self.test_res['test_predict_img']))

        # Create a link to the image
        photo_author = self.test_res['test_img_info']['Author']
        photo_source = self.test_res['test_img_info']['Source']
        photo_source_link = 'https://{0}.com'.format(photo_source.lower())
        photo_number = self.test_res['test_img_info']['Name'].split('_', maxsplit=1)[0]
        photo_license = self.test_res['test_img_info']['License'].split(')')[0].split('(')[1]
        current.card.append(Markdown(
            "Photo by {0} on [{1}]({2}). № {3}. License: {4} \
            *The photo modified: boxes and scores drawn*.".format(
                photo_author, photo_source, photo_source_link, photo_number, photo_license)))
        self.next(self.end)

    @step
    def end(self):
        """Finish"""
        print("Flow is done!")


if __name__ == '__main__':
    MLWorkFlow()

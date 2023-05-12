## Project Directory Structure
---

> Designations used:
> - `{}` - a directory/file name can be changed using a configuration file
> - `<==` - a configuration file and a specific parameter in it to set a new directory/file name or an explanation of how it is generated
> - `*` - (if in a file name) the number of such files can be any
> - `/*` - a directory contains various files and subdirectories.

```
.
|-- .dvc/*
|
|-- .github/workflows
|           |-- ci-tests.yml
|           \-- release.yml
|
|-- .metaflow/*
|
|-- .streamlit/config.toml
|
|-- configs
|   |-- {best_params.yaml}   <== configs/params.yaml: hyperparameter_optimization.save_best_parameters_path
|   \-- params.yaml
|
|-- data
|   |-- new
|   |   |-- bboxes/new_bounding_boxes.csv
|   |   |-- images/*.jpg
|   |   \-- new_image_info.csv
|   |
|   |-- prepared
|   |   |-- test.csv
|   |   \-- train.csv
|   |
|   |-- raw
|   |   |-- bboxes/bounding_boxes.csv
|   |   |-- images/*.jpg
|   |   \-- image_info.csv
|   |
|   |-- .gitignore
|   \-- raw.dvc
|
|-- data_checks
|   |-- data_check_results
|   |   |-- new_csv_file_check_results.json
|   |   |-- new_old_bbox_check_results.html
|   |   |-- new_old_info_check_results.html
|   |   |-- prepared_train_test_author_leakage_check_results.html
|   |   |-- prepared_train_test_similarity_check_results.html
|   |   |-- raw_bbox_duplicates_check_results.html
|   |   \-- raw_csv_file_check_results.json
|   |
|   |-- __init__.py
|   |-- check_bbox_duplicates_and_two_dataset_similarity.py
|   |-- check_img_info_and_bbox_csv_file_integrity.py
|   \-- dch_utils.py
|
|-- deployment
|   |-- demo
|   |   |-- .streamlit/config.toml
|   |   |-- app_demo.py
|   |   |-- fine_tuned_faster_rcnn_mob_large_demo.pt
|   |   |-- README.md
|   |   \-- requirements.txt
|   |
|   |-- static/detected_36485871561.png
|   |
|   |-- api.py
|   \-- app.py
|
|-- docs
|   |-- app-image.pdf
|   |-- dataset-card.md
|   |-- model-card.md
|   |-- project-directory-structure.md
|   |-- project-mlops-diagram.svg
|   \-- project-mlops-diagram-extended.svg
|
|-- great_expectations/*
|
|-- {hyper_opt}                             <== configs.params.yaml: hyperparameter_optimization.save_study_dir
|   |-- {faster_rcnn_mob_hyper_opt_study}   <== configs.params.yaml: hyperparameter_optimization.study_name
|   |   |-- plots
|   |   |   |-- contour.jpg
|   |   |   |-- edf.jpg
|   |   |   |-- intermediate_values.jpg
|   |   |   |-- optimization_history.jpg
|   |   |   |-- parallel_coordinate.jpg
|   |   |   |-- param_importances.jpg
|   |   |   \-- slice.jpg
|   |   |
|   |   \-- .gitignore
|   |
|   \-- hyper_opt_studies.db
|
|-- metaflow_card_cache/*
|
|-- {mlruns}            <== configs.params.yaml: mlflow_tracking_conf.mltracking_uri
|   |-- .trash/*
|   |-- {artifacts}/*   <== configs.params.yaml: mlflow_tracking_conf.artifact_location
|   \-- {mlruns.db}     <== configs.params.yaml: mlflow_tracking_conf.mltracking_uri
|
|-- {models}                                              <== configs.params.yaml: object_detection_model.save_dir
|   \-- {faster_rcnn_mob_best_f_beta_2_weights_ckpt.pt}   <== configs.params.yaml: object_detection_model.name + '_best_' + model_training_inference_conf.metric_to_find_best + '_' + model_training_inference_conf.evaluation_beta + '_weights'(+ '_ckpt' if model_training_inference_conf.save_best_ckpt) + '.pt'
|
|-- {monitoring}                                <== configs.params.yaml: deployed_model_monitoring.save_monitoring_check_results_dir
|   |-- {data/deployed_model_performance.csv}   <== configs.params.yaml: deployed_model_monitoring.save_monitoring_data_path
|   |-- deployed_model_check_results/model_decay_check_results.html
|   |-- __init__.py
|   |-- {current_deployed_model.yaml}           <== configs.params.yaml: deployed_model_monitoring.save_deployed_model_info_path
|   |-- mon_utils.py
|   \-- monitor_deployed_model.py
|
|-- notebooks
|   |-- Dataset_Checks.ipynb
|   |-- EDA.ipynb
|   |-- Fine_Tuning_ObjDet_Model_PyTorch_Colab.ipynb
|   \-- Image_vs_CSV_Dataset_Checks.ipynb
|
|-- {outputs}                            <== configs.params.yaml: model_training_inference_conf.save_model_output_dir
|   |-- plots
|   |   |-- eda
|   |   |   |-- train_author_distribution.jpg
|   |   |   |-- train_avg_bbox_sizes.jpg
|   |   |   |-- train_img_sizes.jpg
|   |   |   \-- train_number_hsparrows_distribution.jpg
|   |   |
|   |   \-- metrics
|   |       |-- {f_beta_2}.jpg           <== configs.params.yaml: model_training_inference_conf.metrics_to_plot + '_' + model_training_inference_conf.evaluation_beta
|   |       \-- {train_epoch_loss}.jpg   <== configs.params.yaml: model_training_inference_conf.metrics_to_plot
|   |
|   |-- test_outs
|   |   |-- predict-*.jpg      <== test.csv: Name (instead of '*')
|   |   \-- test_score.json
|   |
|   |-- val_outs/epoch_*.jpg   <== epoch number (instead of '*')
|   |
|   \-- .gitignore
|
|-- pipelines
|   |-- new_data_pipeline
|   |   |-- dvc.lock
|   |   \-- dvc.yaml
|   |
|   |-- dvc_dag.md
|   |-- dvc.lock
|   |-- dvc.yaml
|   \-- mlworkflow.py
|
|-- {pyttmp}/*                <== pytest.ini: --basetemp (addopts)
|
|-- reports
|   |-- {coverage_report}/*   <== pytest.ini: --cov-report (addopts)
|   |-- deployed_model_performance_report.html
|   |-- model_report.md
|   \-- {test_report.xml}     <== pytest.ini: --junit-xml (addopts)
|
|-- requirements
|   |-- ci-requirements.txt
|   |-- data-check-requirements.txt
|   |-- deployment-requirements.txt
|   |-- dev-requirements.txt
|   |-- eda-requirements.txt
|   |-- monitoring-requirements.txt
|   |-- test-requirements.txt
|   \-- train-requirements.txt
|
|-- src
|   |-- data
|   |   |-- __init__.py
|   |   |-- image_dataloader.py
|   |   |-- prepare_data.py
|   |   \-- update_raw_data.py
|   |
|   |-- model
|   |   |-- __init__.py
|   |   |-- generate_model_report.py
|   |   |-- object_detection_model.py
|   |   \-- update_model_stages.py
|   |
|   |-- train
|   |   |-- __init__.py
|   |   |-- fine_tune_model.py
|   |   |-- model_test_performance.py
|   |   |-- optimize_hyperparams.py
|   |   \-- train_inference_fns.py
|   |
|   |-- __init__.py
|   \-- utils.py
|
|-- tests
|   |-- data_samples
|   |   |-- sample_imgs/*.jpg
|   |   |-- example_config.yaml
|   |   |-- sample_bboxes.csv
|   |   |-- sample_img_info.csv
|   |   |-- sample_monitoring_data.csv
|   |   |-- sample_train.csv
|   |   |-- sample_train_val.csv
|   |   \-- sample_val.csv
|   |
|   |-- datacheck
|   |   |-- test_data_check_fns.py
|   |   \-- __init__.py
|   |
|   |-- integration
|   |   |-- __init__.py
|   |   \-- test_module_integration.py
|   |
|   |-- monitoring
|   |   |-- __init__.py
|   |   |-- conftest.py
|   |   \-- test_monitor_deployed_model_fns.py
|   |
|   |-- unit
|   |   |-- __init__.py
|   |   |-- conftest.py
|   |   |-- test_data_and_model_pre_post_processing_fns.py
|   |   |-- test_image_dataloader.py
|   |   |-- test_optimize_hyperparams.py
|   |   |-- test_train_inference_fns.py
|   |   \-- test_utils_fns.py
|   |
|   |-- webapi
|   |   |-- __init__.py
|   |   \-- test_api.py
|   |
|   |-- __init__.py
|   \-- conftest.py
|
|-- .bandit
|-- .coverage
|-- .dvcignore
|-- .flake8
|-- .gitignore
|-- .isort.cfg
|-- .pre-commit-config.yaml
|-- .yamlignore
|-- .yamllint
|-- mon.log
|-- pipe.log
|-- pytest.ini
\-- README.md
```

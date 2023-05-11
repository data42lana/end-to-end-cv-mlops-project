"""This module builds a RESTful API using FastAPI (backend)."""

from pathlib import Path
from typing import List

import mlflow
import pandas as pd
import yaml
from fastapi import BackgroundTasks, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.train.train_inference_fns import predict
from src.utils import (collate_batch, get_device, get_latest_registered_pytorch_model,
                       get_param_config_yaml)

DEFAULT_PATHS = {'project_path': Path.cwd()}
MODEL_PARAMS = {}

# Define allowed origins
origins = [
    'http://localhost:8501',
]


class DetectionResult(BaseModel):
    """An output response to a detection request."""

    boxes: List[List[float]]
    scores: List[float]
    labels: List[int]


app = FastAPI(title="HouseSparrowDetectionAPI",
              description="The API helps to detect house sparrows in a photo.",
              version="1.0.0")

# Set Cross-Domain permissions
app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=['POST', 'GET'])


def track_predict_result(image, prediction_result, filepath):
    """Log image (photo) size and a model prediction result."""
    tracked_data = {'labels': prediction_result['labels'],
                    'bbox_score': prediction_result['scores']}
    tracked_data.update({f'bbox_{k}': v for k, v in zip(['x1', 'y1', 'x2', 'y2'],
                                                        collate_batch(prediction_result['boxes']))})
    tracked_data.update(
        {k: [v] * len(tracked_data['bbox_score']) for k, v in zip(
            ['reg_model_name', 'reg_model_version', 'image_width', 'image_height'],
            [MODEL_PARAMS['reg_model_name'], MODEL_PARAMS['reg_model_version'],
             image.size[0], image.size[1]])})
    track_df = pd.DataFrame(tracked_data)

    # Save the tracked data
    write_header = False
    if not filepath.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
        write_header = True
    track_df.to_csv(filepath, index=False, mode='a', header=write_header)


@app.on_event('startup')
def set_model_monitoring_configs():
    """Get the configs for model prediction and monitoring."""
    params = get_param_config_yaml(DEFAULT_PATHS['project_path'])
    MODEL_PARAMS['reg_model_name'] = params['object_detection_model']['registered_name']
    MODEL_PARAMS['device'] = get_device(params['model_training_inference_conf']['device_cuda'])
    MODEL_PARAMS['mltracking_uri'] = params['mlflow_tracking_conf']['mltracking_uri']

    monitoring_params = params['deployed_model_monitoring']
    DEFAULT_PATHS['path_to_save_pred_result'] = monitoring_params['save_monitoring_data_path']
    DEFAULT_PATHS['path_to_save_deployed_model_info'] = monitoring_params['save_deployed_model_info_path']  # noqa: B950


@app.on_event('startup')
def init_model():
    """Get the latest model registered in MLflow."""
    mlflow.set_tracking_uri(MODEL_PARAMS['mltracking_uri'])
    client = mlflow.MlflowClient()
    MODEL_PARAMS['model'], model_uri = get_latest_registered_pytorch_model(
        client, MODEL_PARAMS['reg_model_name'],
        stages=['Production'], device=MODEL_PARAMS['device'])
    MODEL_PARAMS['reg_model_version'] = int(model_uri.split('/')[-1])


@app.on_event('startup')
def save_deployed_model_name_version():
    """Save a deployed model name and version."""
    save_deployed_model_info_path = (
        DEFAULT_PATHS['project_path'] / DEFAULT_PATHS['path_to_save_deployed_model_info'])
    save_deployed_model_info_path.parent.mkdir(exist_ok=True, parents=True)

    with open(save_deployed_model_info_path, 'w') as f:
        yaml.safe_dump({'registered_model_name': MODEL_PARAMS['reg_model_name'],
                        'registered_model_version': MODEL_PARAMS['reg_model_version']}, f)


@app.get('/')
def get_root():
    """Return a welcome message."""
    return {'message':
            "Welcome to the House Sparrow Detection API! "
            "See API documentation at /docs and /redoc."}


@app.post('/detection', response_model=DetectionResult)
def make_prediction(input_image: UploadFile, background_tasks: BackgroundTasks):
    """Return an object detection model inference."""
    img = Image.open(input_image.file).convert('RGB')
    result = predict(img, MODEL_PARAMS['model'], MODEL_PARAMS['device'])
    result = {k: v.tolist() for k, v in result.items()}
    background_tasks.add_task(
        track_predict_result, img, result,
        DEFAULT_PATHS['project_path'] / DEFAULT_PATHS['path_to_save_pred_result'])
    return result


if __name__ == '__main__':
    import uvicorn

    # Use reload=True for development only
    # See API documentation at /docs and /redoc
    uvicorn.run('api:app', host='127.0.0.1', port=8000,
                reload=False, reload_dirs=['deployment'])

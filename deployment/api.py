"""This module builds a RESTful API using FastAPI (backend)."""

from pathlib import Path
from typing import List

import mlflow
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.train.train_inference_fns import predict
from src.utils import (get_device, get_latest_registered_pytorch_model,
                       get_param_config_yaml)

PROJECT_PATH = Path.cwd()
MODEL_PARAMS = {}

# Define allowed origins
origins = [
    'http://localhost:8501',
]


class DetectionResult(BaseModel):
    """Output response for a detection request."""

#     model_uri: str
    boxes: List[List[float]]
    scores: List[float]
    labels: List[int]


app = FastAPI(title="HouseSparrowDetectionAPI",
              description="The API helps to detect house sparrows in a image.",
              version="1.0.0")

# Set Cross-Domain permissions
app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=['POST', 'GET'])


@app.on_event('startup')
def set_model_params():
    """Get parameters for model prediction."""
    params = get_param_config_yaml(PROJECT_PATH)
    MODEL_PARAMS['reg_model_name'] = params['object_detection_model']['registered_name']
    MODEL_PARAMS['device'] = get_device(params['model_training_inference_conf']['device_cuda'])
    MODEL_PARAMS['mltracking_uri'] = params['mlflow_tracking_conf']['mltracking_uri']


@app.on_event('startup')
def init_model():
    """Get the latest model registered in MLflow"""
    mlflow.set_tracking_uri(MODEL_PARAMS['mltracking_uri'])
    client = mlflow.MlflowClient()
    MODEL_PARAMS['model'], _ = get_latest_registered_pytorch_model(
        client, MODEL_PARAMS['reg_model_name'],
        stages=['Production'], device=MODEL_PARAMS['device'])


@app.get('/')
def get_root():
    """Return a welcome message."""
    return {'message':
            "Welcome to the House Sparrow Detection API! "
            "The API documentation served at /docs and /redoc."}


@app.post('/detection', response_model=DetectionResult)
def make_prediction(input_image: UploadFile):
    """Return an object detection model inference."""
    img = Image.open(input_image.file).convert('RGB')
    result = predict(img, MODEL_PARAMS['model'], MODEL_PARAMS['device'])
    result = {k: v.tolist() for k, v in result.items()}
    return result


if __name__ == '__main__':
    import uvicorn

    # Use reload=True for development only
    # Documentations served at /docs and /redoc
    uvicorn.run('api:app', host='127.0.0.1', port=8000,
                reload=True, reload_dirs=['deployment'])

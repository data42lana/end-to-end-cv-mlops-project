import pytest
import requests
from fastapi.testclient import TestClient

from deployment.api import app  # isort: split

pytestmark = [pytest.mark.smoke]


class TestFastAPIBackend:

    def test_get_root(self):
        welcom_msg = {'message': 'Welcome to the House Sparrow Detection API! '
                                 'The API documentation served at /docs and /redoc.'}
        response = requests.get('http://127.0.0.1:8000/')
        assert response.status_code == 200
        assert response.json() == welcom_msg

    def test_make_prediction(self, imgs_path):
        # Use TestClient as a context manager to call a startup event
        with TestClient(app) as client:
            img_path = next(imgs_path.iterdir())
            img_file = {'input_image': open(img_path, 'rb')}
            response = client.post('http://127.0.0.1:8000/detection', files=img_file)
        assert response.status_code == 200
        assert sorted(response.json().keys()) == ['boxes', 'labels', 'scores']

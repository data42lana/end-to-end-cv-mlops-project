import pytest
import requests
from fastapi.testclient import TestClient

from deployment.api import app, DEFAULT_PATHS  # isort: split

pytestmark = pytest.mark.web


class TestFastAPIBackend:

    def test_get_root(self):
        welcome_msg = {'message': 'Welcome to the House Sparrow Detection API! '
                                  'See API documentation at /docs and /redoc.'}
        response = requests.get('http://127.0.0.1:8000/')
        assert response.status_code == 200
        assert response.json() == welcome_msg

    def test_make_prediction(self, imgs_path, monkeypatch, tmp_path):
        # Use TestClient as a context manager to call startup events
        with TestClient(app) as client:
            img_path = next(imgs_path.iterdir())
            img_file = {'input_image': open(img_path, 'rb')}
            monkeypatch.setitem(DEFAULT_PATHS, 'project_path', tmp_path)
            response = client.post('http://127.0.0.1:8000/detection', files=img_file)
        assert response.status_code == 200
        assert sorted(response.json().keys()) == ['boxes', 'labels', 'scores']
        assert (tmp_path / DEFAULT_PATHS['path_to_save_pred_result']).exists()

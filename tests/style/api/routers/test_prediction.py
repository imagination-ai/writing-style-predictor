from unittest import mock

from fastapi.testclient import TestClient
from style.api.routers.prediction import router

client = TestClient(router)


def test_prediction_router():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "message": "Predictions Router is working!",
    }


def test_predict():
    response = client.get("/predict?text=sdafas&model_name=small")
    assert response.status_code == 200
    assert response.json()["success"]

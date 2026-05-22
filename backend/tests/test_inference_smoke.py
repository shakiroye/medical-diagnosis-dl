from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


ONE_BY_ONE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAA" 
    "AAC0lEQVR42mP8/5+hHgAFgwJ/lD7xYQAAAABJRU5ErkJggg=="
)


def test_ready_reports_loaded_models() -> None:
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["models"]["tabular"] is True
    assert body["models"]["text"] is True
    assert body["models"]["image"] is True
    assert body["models"]["multimodal"] is True
    assert body["models"]["tabular_loaded"] is True
    assert body["models"]["text_loaded"] is True
    assert body["models"]["image_loaded"] is True
    assert body["models"]["multimodal_loaded"] is True


def test_predict_tabular_uses_loaded_model() -> None:
    response = client.post(
        "/api/v1/predict/tabular",
        json={"patient_id": "P001", "features": {"age": 65, "sex": 1, "cp": 3, "chol": 240, "thalach": 150}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "tabular_classifier"
    assert body["modality"] == "tabular"


def test_predict_text_uses_loaded_model() -> None:
    response = client.post(
        "/api/v1/predict/text",
        json={"patient_id": "P002", "text": "Patient reports cough and chest pain with mild dyspnea.", "language": "en"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "text_classifier"
    assert body["modality"] == "text"


def test_predict_image_uses_loaded_model() -> None:
    response = client.post(
        "/api/v1/predict/image",
        json={"patient_id": "P003", "image_base64": ONE_BY_ONE_PNG_BASE64, "image_type": "chest_xray"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "image_classifier"
    assert body["modality"] == "image"


def test_predict_multimodal_uses_loaded_model() -> None:
    response = client.post(
        "/api/v1/predict/multimodal",
        json={
            "patient_id": "P004",
            "tabular": {"patient_id": "P004", "features": {"age": 58, "sex": 0, "cp": 2, "chol": 210, "thalach": 140}},
            "text": {"patient_id": "P004", "text": "Short report with fever and cough.", "language": "en"},
            "image": {"patient_id": "P004", "image_base64": ONE_BY_ONE_PNG_BASE64, "image_type": "chest_xray"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "multimodal_fusion"
    assert body["modality"] == "multimodal"

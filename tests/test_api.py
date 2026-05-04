from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_flood_prediction_returns_expected_fields():
    response = client.post(
        "/predict/flood",
        json={
            "rainfall": 80,
            "temperature": 25,
            "humidity": 70,
            "wind_speed": 10,
            "ndvi": 0.4,
            "elevation": 100,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["risk_probability"] <= 1
    assert "intensity" in data
    assert data["risk_label"] in {"Low Risk", "Moderate Risk", "High Risk"}
    assert 0 <= data["confidence"] <= 1


def test_wildfire_prediction_returns_expected_fields():
    response = client.post(
        "/predict/wildfire",
        json={
            "rainfall": 20,
            "temperature": 35,
            "humidity": 25,
            "wind_speed": 18,
            "ndvi": 0.2,
            "elevation": 300,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["risk_probability"] <= 1
    assert "intensity" in data
    assert data["risk_label"] in {"Low Risk", "Moderate Risk", "High Risk"}
    assert 0 <= data["confidence"] <= 1

from egov_bot.app import create_test_app
from egov_bot.config import Settings


def test_health_contract(tmp_path):
    settings = Settings(cache_dir=tmp_path, data_dir=tmp_path, sqlite_path=tmp_path / "test.db")
    app = create_test_app(settings=settings)
    client = app.test_client()

    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert "version" in data


def test_chat_contract_without_model(tmp_path):
    settings = Settings(cache_dir=tmp_path, data_dir=tmp_path, sqlite_path=tmp_path / "test.db")
    app = create_test_app(settings=settings)
    client = app.test_client()

    response = client.post("/chat", json={"question": "Đăng ký khai sinh cần gì?", "session_id": "test"})
    assert response.status_code == 200
    data = response.get_json()
    assert {"answer", "sources", "request_id", "latency_ms", "cached", "context_source"}.issubset(data)


def test_feedback_contract(tmp_path):
    settings = Settings(cache_dir=tmp_path, data_dir=tmp_path, sqlite_path=tmp_path / "test.db")
    app = create_test_app(settings=settings)
    client = app.test_client()

    response = client.post("/feedback", json={"rating": "like", "session_id": "test"})
    assert response.status_code == 200
    assert response.get_json()["summary"]["likes"] == 1


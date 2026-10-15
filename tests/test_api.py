"""API tests with a stand-in model, so they run in CI without downloading weights."""

import pytest
from fastapi.testclient import TestClient

from api.main import MAX_TEXTS, app


class FakeModel:
    source = "fake-model"

    def classify(self, texts):
        return [
            {"text": t, "sentiment": "positive" if "love" in t else "negative", "confidence": 0.9}
            for t in texts
        ]


@pytest.fixture
def client():
    app.state.model = FakeModel()
    with TestClient(app) as c:
        yield c
    app.state.model = None


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model": "fake-model"}


def test_predict_returns_one_prediction_per_text_in_order(client):
    r = client.post("/predict", json={"texts": ["I love this", "worst day ever"]})
    assert r.status_code == 200
    body = r.json()
    assert [p["sentiment"] for p in body["predictions"]] == ["positive", "negative"]
    assert body["model"] == "fake-model"
    assert body["elapsed_ms"] >= 0


@pytest.mark.parametrize(
    "payload",
    [{}, {"texts": []}, {"texts": "not a list"}, {"texts": ["x"] * (MAX_TEXTS + 1)}],
)
def test_predict_rejects_bad_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_long_text_is_truncated_not_rejected(client):
    r = client.post("/predict", json={"texts": ["a" * 5000]})
    assert r.status_code == 200
    assert len(r.json()["predictions"][0]["text"]) == 1000


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/docs"

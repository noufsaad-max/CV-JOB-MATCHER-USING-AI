"""Tests for the /compare endpoint.

Heavy dependencies (gensim, nltk, PyPDF2) are stubbed so the tests run
without the trained model or a network connection.
"""
import io
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return sys.modules[name]


_stub("PyPDF2", PdfReader=object)
_stub("gensim")
_stub("gensim.models", Doc2Vec=object, FastText=object)
_stub("gensim.models.doc2vec", TaggedDocument=object)
_stub("nltk", download=lambda *a, **k: None)
_stub("flask_cors", CORS=lambda app: None)

import app as app_module  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "UPLOAD_FOLDER", str(tmp_path))
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client(), tmp_path


def _post(c):
    data = {
        "file1": (io.BytesIO(b"%PDF a"), "cv.pdf"),
        "file2": (io.BytesIO(b"%PDF b"), "job.pdf"),
    }
    return c.post("/compare", data=data, content_type="multipart/form-data")


def test_missing_file_returns_400(client):
    c, _ = client
    resp = c.post("/compare", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_pdf_without_text_returns_clear_400_and_cleans_up(client, monkeypatch):
    c, upload_dir = client
    texts = iter(["Python developer with Flask", "   "])
    monkeypatch.setattr(app_module, "extract_text_from_pdf", lambda p: next(texts))

    resp = _post(c)

    assert resp.status_code == 400
    assert "text" in resp.get_json()["error"].lower()
    assert os.listdir(upload_dir) == []


def test_success_returns_rounded_score_and_cleans_up(client, monkeypatch):
    c, upload_dir = client
    monkeypatch.setattr(app_module, "extract_text_from_pdf", lambda p: "some text")
    monkeypatch.setattr(app_module, "calculate_similarity", lambda a, b: 87.4567)

    resp = _post(c)

    assert resp.status_code == 200
    assert resp.get_json() == {"score": 87.46}
    assert os.listdir(upload_dir) == []


def test_error_returns_500_and_cleans_up(client, monkeypatch):
    c, upload_dir = client
    monkeypatch.setattr(app_module, "extract_text_from_pdf", lambda p: "some text")

    def boom(a, b):
        raise RuntimeError("model file missing")

    monkeypatch.setattr(app_module, "calculate_similarity", boom)

    resp = _post(c)

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "model file missing"
    assert os.listdir(upload_dir) == []

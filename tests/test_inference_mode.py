import asyncio
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from src.api import server
from src.core.classifier import MicrostructureClassifier


def test_classifier_resets_trained_model_when_switching_to_model():
    classifier = MicrostructureClassifier(inference_mode="heuristic")
    hmm_model = Mock()
    classifier.hmm_model = hmm_model

    assert classifier.set_inference_mode("model") == "model"
    hmm_model.reset.assert_called_once_with()


def test_classifier_does_not_claim_model_mode_when_artifact_is_missing():
    classifier = MicrostructureClassifier(inference_mode="heuristic")
    classifier.hmm_model = None

    with patch.object(classifier, "_load_hmm_model"):
        with pytest.raises(RuntimeError, match="No hay un modelo HMM entrenado"):
            classifier.set_inference_mode("model")

    assert classifier.inference_mode == "heuristic"


def test_classifier_falls_back_to_heuristic_on_startup_without_model():
    with patch.object(MicrostructureClassifier, "_load_hmm_model"):
        classifier = MicrostructureClassifier(inference_mode="model")

    assert classifier.hmm_model is None
    assert classifier.inference_mode == "heuristic"


def test_http_toggle_endpoint_returns_the_changed_mode():
    fake_classifier = Mock()
    fake_classifier.toggle_inference_mode.return_value = "heuristic"

    with patch.object(server, "classifier", fake_classifier):
        response = asyncio.run(server.toggle_inference_mode_endpoint())

    assert response.status_code == 200
    assert response.body == b'{"status":"ok","inference_mode":"heuristic"}'


def test_http_toggle_endpoint_reports_missing_trained_model():
    fake_classifier = Mock()
    fake_classifier.toggle_inference_mode.side_effect = RuntimeError(
        "No hay un modelo HMM entrenado disponible para cargar."
    )

    with patch.object(server, "classifier", fake_classifier):
        with pytest.raises(HTTPException) as error:
            asyncio.run(server.toggle_inference_mode_endpoint())

    assert error.value.status_code == 503
    assert "modelo HMM entrenado" in error.value.detail

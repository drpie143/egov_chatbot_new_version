from __future__ import annotations

import time

from flask import Blueprint, current_app, jsonify

from egov_bot import __version__

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health():
    settings = current_app.extensions["egov_settings"]
    resources = current_app.extensions["egov_resources"]
    generator = current_app.extensions["egov_generator"]
    return jsonify(
        {
            "status": "ok" if resources.loaded else "degraded",
            "version": __version__,
            "timestamp": time.time(),
            "app_env": settings.app_env,
            "data_source": settings.data_source,
            "procedures_loaded": len(resources.procedures),
            "faiss_loaded": resources.faiss_index is not None,
            "bm25_loaded": resources.bm25 is not None,
            "embedding_model_loaded": resources.embedding_model is not None,
            "generation_model_loaded": generator.available,
            "load_errors": resources.load_errors or [],
        }
    )


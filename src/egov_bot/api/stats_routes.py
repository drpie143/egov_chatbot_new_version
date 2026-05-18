from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

stats_bp = Blueprint("stats", __name__)


@stats_bp.get("/stats/popular")
def popular():
    try:
        limit = int(request.args.get("limit") or 10)
    except ValueError:
        limit = 10
    repository = current_app.extensions["egov_repository"]
    return jsonify({"popular_procedures": repository.popular(limit=max(1, min(limit, 50)))})


@stats_bp.get("/stats/feedback")
def feedback_stats():
    repository = current_app.extensions["egov_repository"]
    return jsonify({"feedback_summary": repository.feedback_summary()})


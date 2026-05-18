from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

feedback_bp = Blueprint("feedback", __name__)


@feedback_bp.post("/feedback")
def feedback():
    payload = request.get_json(silent=True) or {}
    rating = str(payload.get("rating") or payload.get("new_feedback") or "").lower()
    if rating in {"", "none", "null"}:
        rating = "neutral"
    request_id = payload.get("request_id")
    session_id = payload.get("session_id")
    comment = payload.get("comment") or payload.get("message")
    repository = current_app.extensions["egov_repository"]
    try:
        repository.save_feedback(
            rating=rating,
            request_id=str(request_id) if request_id else None,
            session_id=str(session_id) if session_id else None,
            comment=str(comment) if comment else None,
        )
    except ValueError as exc:
        return jsonify({"error": True, "message": str(exc)}), 400
    return jsonify({"status": "success", "summary": repository.feedback_summary()})


@feedback_bp.post("/save_feedback")
def legacy_save_feedback():
    return feedback()


@feedback_bp.post("/update_popular")
def legacy_update_popular():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name") or "").strip()
    url = str(payload.get("url") or "").strip()
    if not name:
        return jsonify({"error": True, "message": "Missing procedure name."}), 400
    repository = current_app.extensions["egov_repository"]
    repository.increment_popular(name, url)
    return jsonify({"status": "success", "name": name})


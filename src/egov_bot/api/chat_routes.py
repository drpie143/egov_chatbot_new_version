from __future__ import annotations

import logging
import uuid

from flask import Blueprint, current_app, jsonify, request

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__)


@chat_bp.post("/chat")
def chat():
    request_id = str(uuid.uuid4())
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": True, "message": "Invalid JSON payload.", "request_id": request_id}), 400

    question = str(payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": True, "message": "Missing question.", "request_id": request_id}), 400

    session_id = str(payload.get("session_id") or "default")
    pipeline = current_app.extensions["egov_pipeline"]
    try:
        result = pipeline.answer(question=question, session_id=session_id, request_id=request_id)
        return jsonify(result.to_dict())
    except Exception as exc:
        logger.exception("Chat request failed request_id=%s", request_id)
        return (
            jsonify(
                {
                    "error": True,
                    "message": "Hệ thống đang gặp sự cố khi tạo câu trả lời. Vui lòng thử lại sau.",
                    "request_id": request_id,
                    "detail": str(exc) if current_app.debug else None,
                }
            ),
            500,
        )


@chat_bp.post("/clear_session")
def clear_session():
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("session_id") or "default")
    sessions = current_app.extensions["egov_sessions"]
    existed = sessions.clear(session_id)
    return jsonify({"status": "success", "session_id": session_id, "cleared": existed})


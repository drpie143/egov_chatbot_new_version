from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

search_bp = Blueprint("search", __name__)


@search_bp.get("/search")
def search():
    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"query": query, "results": []})
    try:
        limit = int(request.args.get("limit") or current_app.extensions["egov_settings"].search_limit)
    except ValueError:
        limit = current_app.extensions["egov_settings"].search_limit
    limit = max(1, min(limit, 50))

    retriever = current_app.extensions["egov_retriever"]
    results = retriever.search(query, limit=limit)
    return jsonify({"query": query, "results": [source.to_dict() for source in results]})


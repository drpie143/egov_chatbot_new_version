from __future__ import annotations

from flask import Flask

from egov_bot.api.chat_routes import chat_bp
from egov_bot.api.feedback_routes import feedback_bp
from egov_bot.api.health_routes import health_bp
from egov_bot.api.search_routes import search_bp
from egov_bot.api.stats_routes import stats_bp


def register_blueprints(app: Flask) -> None:
    app.register_blueprint(health_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(stats_bp)


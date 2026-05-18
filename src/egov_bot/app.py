from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, abort, render_template, send_from_directory
from flask_cors import CORS

from egov_bot.api import register_blueprints
from egov_bot.config import Settings, load_settings
from egov_bot.conversation.session_manager import SessionManager
from egov_bot.data.procedure_store import ProcedureStore
from egov_bot.data.resource_loader import Resources, load_resources
from egov_bot.logging_config import configure_logging
from egov_bot.rag.answer_generator import GeminiAnswerGenerator
from egov_bot.rag.pipeline import RAGPipeline
from egov_bot.retrieval.hybrid_retriever import HybridRetriever
from egov_bot.storage.db import Database
from egov_bot.storage.repositories import AppRepository

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, load_model_resources: bool = True) -> Flask:
    settings = settings or load_settings()
    configure_logging(settings.log_level)

    app = Flask(
        __name__,
        template_folder=str(Path.cwd() / "templates"),
        static_folder=str(Path.cwd() / "static"),
    )
    app.config["JSON_AS_ASCII"] = False
    app.config["APP_ENV"] = settings.app_env
    CORS(app, origins=settings.cors_origins or "*")

    resources = load_resources(settings, load_models=load_model_resources)
    db = Database(settings.sqlite_path)
    db.import_legacy_json(Path("user_data/popular_procedures.json"), Path("user_data/user_feedback.json"))
    repository = AppRepository(db)
    sessions = SessionManager()
    generator = GeminiAnswerGenerator(settings)
    retriever = HybridRetriever(
        settings=settings,
        procedure_store=resources.procedure_store,
        metadatas=resources.metadatas,
        faiss_index=resources.faiss_index,
        bm25=resources.bm25,
        embedding_model=resources.embedding_model,
    )
    pipeline = RAGPipeline(
        settings=settings,
        procedure_store=resources.procedure_store,
        retriever=retriever,
        sessions=sessions,
        repository=repository,
        generator=generator,
    )

    app.extensions["egov_settings"] = settings
    app.extensions["egov_resources"] = resources
    app.extensions["egov_repository"] = repository
    app.extensions["egov_sessions"] = sessions
    app.extensions["egov_generator"] = generator
    app.extensions["egov_retriever"] = retriever
    app.extensions["egov_pipeline"] = pipeline

    register_blueprints(app)

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/user_data/<path:filename>")
    def user_data_files(filename: str):
        user_data_dir = Path.cwd() / "user_data"
        requested_path = (user_data_dir / filename).resolve()
        root = user_data_dir.resolve()
        if requested_path != root and root not in requested_path.parents:
            abort(404)
        if not requested_path.is_file():
            abort(404)
        return send_from_directory(user_data_dir, filename, as_attachment=False)

    logger.info("eGov app created with %s procedures", len(resources.procedures))
    return app


def create_test_app(settings: Settings | None = None, resources: Resources | None = None) -> Flask:
    settings = settings or load_settings()
    configure_logging(settings.log_level)
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["JSON_AS_ASCII"] = False
    CORS(app, origins="*")

    if resources is None:
        empty_store = ProcedureStore([])
        resources = Resources(procedure_store=empty_store, procedures=[], metadatas=[])

    db = Database(settings.sqlite_path)
    repository = AppRepository(db)
    sessions = SessionManager()
    generator = GeminiAnswerGenerator(settings)
    retriever = HybridRetriever(
        settings=settings,
        procedure_store=resources.procedure_store,
        metadatas=resources.metadatas,
        faiss_index=resources.faiss_index,
        bm25=resources.bm25,
        embedding_model=resources.embedding_model,
    )
    pipeline = RAGPipeline(settings, resources.procedure_store, retriever, sessions, repository, generator)

    app.extensions["egov_settings"] = settings
    app.extensions["egov_resources"] = resources
    app.extensions["egov_repository"] = repository
    app.extensions["egov_sessions"] = sessions
    app.extensions["egov_generator"] = generator
    app.extensions["egov_retriever"] = retriever
    app.extensions["egov_pipeline"] = pipeline
    register_blueprints(app)
    return app


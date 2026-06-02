from __future__ import annotations

import json
import sys
from http import HTTPStatus
from pathlib import Path

from flask import Flask, Response, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import bootstrap_payload, recommendations_payload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = PROJECT_ROOT / "web"

app = Flask(__name__, static_folder=None)


@app.route("/api/bootstrap")
def api_bootstrap():
    return Response(
        json.dumps(bootstrap_payload()),
        mimetype="application/json",
    )


@app.route("/api/recommendations")
def api_recommendations():
    try:
        return Response(
            json.dumps(recommendations_payload(dict(request.args.lists()))),
            mimetype="application/json",
        )
    except (KeyError, ValueError) as exc:
        return Response(
            json.dumps({"error": str(exc)}),
            status=HTTPStatus.BAD_REQUEST,
            mimetype="application/json",
        )


@app.route("/")
def index():
    return send_from_directory(str(PUBLIC_DIR), "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(str(PUBLIC_DIR), path)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Iris Recognition System web app for Jetson Nano.

Python 3.6 compatible.
"""

import argparse
import os
import socket
import time
import traceback
import uuid

try:
    from flask import Flask, request, jsonify, render_template
except Exception as e:
    print("ERROR: Flask is not installed.")
    print("Install it on Jetson with:")
    print("  sudo apt update")
    print("  sudo apt install python3-flask -y")
    raise e

import database
from iris_backend import IrisBackend, meta_to_json


APP_TITLE = "Iris Recognition System"
DEFAULT_DB = os.path.join("data", "iris_users_web.json")
DEFAULT_THRESHOLD = 0.30
MAX_UPLOAD_BYTES = 32 * 1024 * 1024


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
# Demo-friendly static assets: avoid stale CSS/JS after quick Jetson updates.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

ARGS = None
BACKEND = None


def now_text():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def elapsed_ms(start):
    return int((time.time() - start) * 1000.0)


def get_local_ip():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.getcwd(), path))


def upload_dir():
    path = os.path.abspath(os.path.join(os.getcwd(), "uploads"))
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def safe_filename(name):
    name = os.path.basename(name or "upload.jpg")
    out = []
    for ch in name:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    name = "".join(out)
    return name or "upload.jpg"


def save_upload(file_obj, prefix):
    if file_obj is None:
        raise RuntimeError("No file uploaded")
    filename = prefix + "_" + uuid.uuid4().hex[:12] + "_" + safe_filename(file_obj.filename)
    path = os.path.join(upload_dir(), filename)
    file_obj.save(path)
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        raise RuntimeError("Uploaded file is empty")
    return path


def parse_threshold(value):
    try:
        threshold = float(value)
    except Exception:
        threshold = float(ARGS.threshold)
    if threshold < 0.0:
        threshold = 0.0
    if threshold > 1.0:
        threshold = 1.0
    return threshold


@app.route("/", methods=["GET"])
def index():
    summary = database.db_summary(ARGS.db)
    ip = get_local_ip()
    return render_template(
        "index.html",
        app_title=APP_TITLE,
        default_threshold="{0:.2f}".format(float(ARGS.threshold)),
        server_port=ARGS.port,
        initial_status="Online",
        initial_user_count=summary["count"],
        initial_server="%s:%d" % (ip, ARGS.port)
    )


@app.route("/api/health", methods=["GET"])
def api_health():
    summary = database.db_summary(ARGS.db)
    return jsonify({
        "ok": True,
        "app": APP_TITLE,
        "time": now_text(),
        "ip": get_local_ip(),
        "port": ARGS.port,
        "db_count": summary["count"],
        "db_users": summary["users"],
        "engine": ARGS.engine,
        "meta": ARGS.meta,
        "db": ARGS.db,
        "threshold": float(ARGS.threshold)
    })


@app.route("/api/users", methods=["GET"])
def api_users():
    users = database.list_users(ARGS.db)
    return jsonify({"ok": True, "count": len(users), "users": users})


@app.route("/api/delete_all", methods=["POST"])
def api_delete_all():
    database.reset_db(ARGS.db)
    return jsonify({"ok": True, "message": "Database cleared"})


@app.route("/api/register", methods=["POST"])
def api_register():
    start = time.time()
    try:
        name = (request.form.get("name") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "Name is required"}), 400

        left_file = request.files.get("left")
        right_file = request.files.get("right")
        if left_file is None:
            return jsonify({"ok": False, "error": "Left eye image is required"}), 400
        if right_file is None:
            return jsonify({"ok": False, "error": "Right eye image is required"}), 400

        left_path = save_upload(left_file, "left")
        right_path = save_upload(right_file, "right")

        t = time.time()
        left_emb, left_meta = BACKEND.make_embedding(left_path)
        left_ms = elapsed_ms(t)

        t = time.time()
        right_emb, right_meta = BACKEND.make_embedding(right_path)
        right_ms = elapsed_ms(t)

        summary = database.register_both(ARGS.db, name, left_emb, right_emb)

        return jsonify({
            "ok": True,
            "message": "Enrolled both eyes for " + name,
            "name": name,
            "elapsed_ms": elapsed_ms(start),
            "left_ms": left_ms,
            "right_ms": right_ms,
            "left_meta": meta_to_json(left_meta),
            "right_meta": meta_to_json(right_meta),
            "db": summary
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "elapsed_ms": elapsed_ms(start)}), 500


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    start = time.time()
    try:
        threshold = parse_threshold(request.form.get("threshold") or ARGS.threshold)
        img_file = request.files.get("image")
        if img_file is None:
            return jsonify({"ok": False, "error": "Iris image is required"}), 400

        img_path = save_upload(img_file, "query")
        t = time.time()
        emb, meta = BACKEND.make_embedding(img_path)
        infer_ms = elapsed_ms(t)
        name, eye, score, top = database.recognize(ARGS.db, emb, threshold, 8)

        return jsonify({
            "ok": True,
            "matched": name is not None,
            "name": name,
            "eye": eye,
            "score": score,
            "threshold": threshold,
            "top_scores": top,
            "elapsed_ms": elapsed_ms(start),
            "inference_ms": infer_ms,
            "meta": meta_to_json(meta)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "elapsed_ms": elapsed_ms(start)}), 500


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="iris_model_fp16.engine", help="TensorRT engine path")
    parser.add_argument("--meta", default="iris_model.metadata.json", help="metadata JSON path")
    parser.add_argument("--db", default=DEFAULT_DB, help="local JSON enrollment database")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="default cosine threshold")
    parser.add_argument("--host", default="0.0.0.0", help="host address")
    parser.add_argument("--port", type=int, default=8000, help="web server port")
    return parser


def main():
    global ARGS, BACKEND
    parser = build_parser()
    ARGS = parser.parse_args()
    ARGS.engine = resolve_path(ARGS.engine)
    ARGS.meta = resolve_path(ARGS.meta)
    ARGS.db = resolve_path(ARGS.db)

    print("")
    print("=" * 72)
    print(APP_TITLE)
    print("=" * 72)
    print("Loading metadata:", ARGS.meta)
    print("Loading TensorRT engine:", ARGS.engine)
    BACKEND = IrisBackend(ARGS.engine, ARGS.meta)
    BACKEND.load()

    ip = get_local_ip()
    print("")
    print("Server ready.")
    print("Open on Jetson:        http://127.0.0.1:%d" % ARGS.port)
    print("Open on local network: http://%s:%d" % (ip, ARGS.port))
    print("Database:", ARGS.db)
    print("Default threshold: %.2f" % float(ARGS.threshold))
    print("=" * 72)
    print("")
    app.run(host=ARGS.host, port=ARGS.port, threaded=False, debug=False)


if __name__ == "__main__":
    main()

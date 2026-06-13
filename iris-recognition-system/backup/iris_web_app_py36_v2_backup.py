#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iris Recognition System - Web UI for Jetson Nano
Python 3.6 compatible.

Requires existing backend:
  ~/iris_app_py36.py

Run:
  cd ~/Documents/thesis
  python3 ~/iris_web_app_py36_v2.py --engine iris_model_fp16.engine --host 0.0.0.0 --port 8000

Open from another device on the same network:
  http://JETSON_IP:8000
"""

import os
import sys
import json
import time
import uuid
import socket
import argparse
import traceback
import threading

import numpy as np

try:
    from flask import Flask, request, jsonify, render_template_string
except Exception as e:
    print("ERROR: Flask is not installed.")
    print("Install it on Jetson with:")
    print("  sudo apt update")
    print("  sudo apt install python3-flask -y")
    raise e

HOME = os.path.expanduser("~")
if HOME not in sys.path:
    sys.path.insert(0, HOME)

try:
    import iris_app_py36 as core
except Exception as e:
    print("ERROR: Could not import ~/iris_app_py36.py")
    print("Make sure iris_app_py36.py exists in your home folder.")
    raise e

APP_TITLE = "Iris Recognition System"
DEFAULT_DB = "iris_users_web.json"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

ARGS = None
CFG = None
INFER = None
INFER_LOCK = threading.Lock()


def now_text():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def upload_dir():
    path = os.path.join("/tmp", "iris_web_uploads")
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
    path = os.path.join(upload_dir(), prefix + "_" + uuid.uuid4().hex[:12] + "_" + safe_filename(file_obj.filename))
    file_obj.save(path)
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        raise RuntimeError("Uploaded file is empty")
    return path


def normalize_vec(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    if n > 0:
        return v / n
    return v


def cosine(a, b):
    return float(np.dot(normalize_vec(a), normalize_vec(b)))


def elapsed_ms(start):
    return int((time.time() - start) * 1000.0)


def meta_to_json(meta):
    if not isinstance(meta, dict):
        return {}
    out = {}
    for k, v in meta.items():
        if isinstance(v, tuple) or isinstance(v, list):
            vals = []
            for x in v:
                try:
                    vals.append(float(x))
                except Exception:
                    vals.append(str(x))
            out[k] = vals
        elif isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        else:
            try:
                json.dumps(v)
                out[k] = v
            except Exception:
                out[k] = str(v)
    return out


def load_db(path):
    if not os.path.exists(path):
        return {"version": 2, "users": {}}
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except Exception:
        return {"version": 2, "users": {}}

    if isinstance(raw, dict) and "users" in raw:
        return raw

    # migrate old format: {"Dat": [embedding...]}
    db = {"version": 2, "users": {}}
    if isinstance(raw, dict):
        for name, emb in raw.items():
            if isinstance(emb, list):
                db["users"][name] = {
                    "created_at": "legacy",
                    "left": emb,
                    "right": None,
                    "notes": "Migrated from old single-eye DB"
                }
    return db


def save_db(path, db):
    with open(path, "w") as f:
        json.dump(db, f, indent=2)


def db_summary():
    db = load_db(ARGS.db)
    users = sorted(list(db.get("users", {}).keys()))
    return {"count": len(users), "users": users}


def register_both(db_path, name, left_emb, right_emb):
    db = load_db(db_path)
    db["users"][name] = {
        "created_at": now_text(),
        "left": normalize_vec(left_emb).tolist(),
        "right": normalize_vec(right_emb).tolist(),
        "notes": "Two-eye enrollment from web UI"
    }
    save_db(db_path, db)


def recognize(db_path, emb, threshold):
    db = load_db(db_path)
    users = db.get("users", {})
    if not users:
        return None, None, None, []

    best_name = None
    best_eye = None
    best_score = -999.0
    top = []

    for name, rec in users.items():
        for eye in ["left", "right"]:
            ref = rec.get(eye)
            if ref is None:
                continue
            ref_arr = np.asarray(ref, dtype=np.float32).reshape(-1)
            ref_norm = float(np.linalg.norm(ref_arr))
            if (not np.isfinite(ref_norm)) or ref_norm < 1e-6:
                top.append({"user": name, "eye": eye, "score": 0.0, "bad_embedding": True})
                continue

            score = cosine(emb, ref_arr)
            top.append({"user": name, "eye": eye, "score": score, "bad_embedding": False})
            if score > best_score:
                best_name = name
                best_eye = eye
                best_score = score

    top.sort(key=lambda x: x["score"], reverse=True)
    if best_score <= -998.0:
        # Every stored embedding was zero/invalid. Return score 0 but mark no usable DB.
        return None, None, 0.0, top[:8]
    if best_score >= threshold:
        return best_name, best_eye, best_score, top[:8]
    return None, best_eye, best_score, top[:8]


def make_embedding(image_path):
    # Serialize access to TensorRT/PyCUDA context for stability.
    with INFER_LOCK:
        polar, meta = core.preprocess_image(image_path, CFG)
        if polar is None:
            reason = "unknown"
            if isinstance(meta, dict):
                reason = meta.get("reason", "unknown")
            raise RuntimeError("Iris segmentation failed: " + str(reason))

        emb = INFER(polar)

    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    emb_norm = float(np.linalg.norm(emb))

    if not np.isfinite(emb_norm) or emb_norm < 1e-6:
        raise RuntimeError(
            "TensorRT returned a zero/invalid embedding. "
            "This usually means the old server hit a CUDA context bug and the current database may contain bad zero embeddings. "
            "Restart this fixed server and re-enroll users after deleting the old DB."
        )

    if isinstance(meta, dict):
        meta["embedding_norm"] = emb_norm
        meta["embedding_dim"] = int(emb.shape[0])

    return emb, meta


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/health", methods=["GET"])
def api_health():
    s = db_summary()
    return jsonify({
        "ok": True,
        "app": APP_TITLE,
        "time": now_text(),
        "ip": get_local_ip(),
        "db_count": s["count"],
        "db_users": s["users"],
        "engine": ARGS.engine,
        "db": ARGS.db
    })


@app.route("/api/users", methods=["GET"])
def api_users():
    db = load_db(ARGS.db)
    users = []
    for name, rec in sorted(db.get("users", {}).items()):
        users.append({
            "name": name,
            "created_at": rec.get("created_at", "unknown"),
            "left": rec.get("left") is not None,
            "right": rec.get("right") is not None,
            "notes": rec.get("notes", "")
        })
    return jsonify({"ok": True, "count": len(users), "users": users})


@app.route("/api/delete_all", methods=["POST"])
def api_delete_all():
    save_db(ARGS.db, {"version": 2, "users": {}})
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
        left_emb, left_meta = make_embedding(left_path)
        left_ms = elapsed_ms(t)

        t = time.time()
        right_emb, right_meta = make_embedding(right_path)
        right_ms = elapsed_ms(t)

        register_both(ARGS.db, name, left_emb, right_emb)

        return jsonify({
            "ok": True,
            "message": "Enrolled both eyes for " + name,
            "name": name,
            "elapsed_ms": elapsed_ms(start),
            "left_ms": left_ms,
            "right_ms": right_ms,
            "left_meta": meta_to_json(left_meta),
            "right_meta": meta_to_json(right_meta),
            "db": db_summary()
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "elapsed_ms": elapsed_ms(start)}), 500


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    start = time.time()
    try:
        threshold = request.form.get("threshold") or str(ARGS.threshold)
        try:
            threshold = float(threshold)
        except Exception:
            threshold = ARGS.threshold

        img_file = request.files.get("image")
        if img_file is None:
            return jsonify({"ok": False, "error": "Iris image is required"}), 400

        img_path = save_upload(img_file, "query")
        t = time.time()
        emb, meta = make_embedding(img_path)
        infer_ms = elapsed_ms(t)
        name, eye, score, top = recognize(ARGS.db, emb, threshold)

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


INDEX_HTML = r'''
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Iris Recognition System</title>
<style>
:root{--bg:#050b18;--panel:#0f172a;--card:#f8fafc;--text:#0f172a;--muted:#64748b;--white:#fff;--blue:#2563eb;--cyan:#06b6d4;--green:#16a34a;--red:#dc2626;--border:rgba(148,163,184,.25);--shadow:0 24px 70px rgba(0,0,0,.38)}
*{box-sizing:border-box} body{margin:0;min-height:100vh;font-family:Inter,system-ui,-apple-system,"Segoe UI",Arial,sans-serif;background:radial-gradient(circle at 15% 12%,rgba(37,99,235,.34),transparent 30%),radial-gradient(circle at 84% 8%,rgba(6,182,212,.25),transparent 28%),linear-gradient(135deg,#020617,#0f172a 60%,#111827);color:white}.shell{width:min(1220px,calc(100% - 32px));margin:auto;padding:24px 0 30px}header{display:flex;justify-content:space-between;align-items:center;gap:18px;margin-bottom:22px}.brand{display:flex;gap:16px;align-items:center}.logo{width:62px;height:62px;border-radius:20px;background:linear-gradient(135deg,var(--blue),var(--cyan));display:grid;place-items:center;box-shadow:0 18px 45px rgba(37,99,235,.38);position:relative}.logo:before{content:"";width:36px;height:18px;border:3px solid #fff;border-radius:50%}.logo:after{content:"";position:absolute;width:13px;height:13px;border-radius:50%;background:#fff}h1{margin:0;font-size:clamp(26px,4vw,40px);letter-spacing:-.05em}.sub{margin-top:4px;color:#94a3b8;font-size:14px}.status{min-width:300px;border:1px solid var(--border);background:rgba(15,23,42,.68);backdrop-filter:blur(18px);border-radius:22px;padding:14px 16px;box-shadow:var(--shadow)}.line{display:flex;justify-content:space-between;color:#cbd5e1;font-size:13px;margin:5px 0}.line b{color:#fff}.tabs{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:18px}.tab{border:1px solid var(--border);background:rgba(15,23,42,.68);color:#cbd5e1;border-radius:999px;padding:12px 16px;cursor:pointer;font-weight:900}.tab.active{color:#fff;background:linear-gradient(135deg,var(--blue),var(--cyan));border-color:transparent}.panel{display:none;border:1px solid var(--border);border-radius:30px;background:rgba(15,23,42,.58);box-shadow:var(--shadow);backdrop-filter:blur(18px);overflow:hidden}.panel.active{display:block}.grid{display:grid;grid-template-columns:1.35fr .85fr;gap:18px;padding:18px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:rgba(248,250,252,.95);color:var(--text);border-radius:24px;padding:22px;box-shadow:0 15px 38px rgba(0,0,0,.15)}.dark{background:linear-gradient(180deg,rgba(15,23,42,.97),rgba(30,41,59,.93));color:#fff;border:1px solid var(--border)}h2{margin:0 0 6px;font-size:26px;letter-spacing:-.04em}h3{margin:0 0 10px;font-size:19px}p{margin:0;color:var(--muted);line-height:1.5}.drop{margin-top:18px;border:2px dashed #cbd5e1;background:#f8fafc;border-radius:24px;min-height:260px;padding:18px;display:grid;place-items:center;text-align:center;transition:.18s;overflow:hidden}.drop.drag{border-color:var(--blue);background:#eff6ff;transform:scale(1.01)}.drop img{display:none;max-width:100%;max-height:330px;border-radius:18px;object-fit:contain;box-shadow:0 14px 35px rgba(15,23,42,.22)}.drop.has img{display:block}.drop.has .hint{display:none}.hintIcon{width:70px;height:70px;border-radius:24px;background:linear-gradient(135deg,#dbeafe,#cffafe);display:grid;place-items:center;color:var(--blue);font-size:34px;font-weight:950;margin:0 auto 12px}.hint strong{display:block;color:#0f172a;font-size:17px;margin-bottom:5px}.hint span{color:#64748b;font-size:13px}input[type=file]{display:none}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:16px}.input{width:100%;border:1px solid #cbd5e1;border-radius:15px;padding:13px 14px;font-size:15px;outline:none;background:#fff;color:#0f172a}.input:focus{border-color:var(--blue);box-shadow:0 0 0 4px rgba(37,99,235,.12)}.btn{border:0;border-radius:16px;padding:13px 16px;cursor:pointer;font-weight:950;color:white;background:var(--blue);box-shadow:0 12px 28px rgba(37,99,235,.22)}.btn:hover{filter:brightness(1.05);transform:translateY(-1px)}.btn.green{background:var(--green)}.btn.red{background:var(--red)}.btn.gray{background:#475569}.decision{font-size:34px;font-weight:950;letter-spacing:-.06em;margin-bottom:5px}.idle{color:#e2e8f0}.match{color:#4ade80}.nomatch{color:#f87171}.result{margin-top:18px;padding:18px;border-radius:22px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12)}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px}.metric{padding:14px;border-radius:18px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.1)}.metric .lab{font-size:12px;color:#94a3b8}.metric .val{font-size:18px;font-weight:950;color:#fff;word-break:break-word}.slider{display:grid;grid-template-columns:1fr 80px;gap:10px;margin-top:12px}.check{display:flex;align-items:center;gap:9px;color:#0f172a;font-weight:900;margin-top:14px}.log{background:#020617;color:#cbd5e1;border-radius:18px;padding:14px;max-height:230px;overflow:auto;font-family:ui-monospace,Consolas,monospace;font-size:12px;margin-top:16px;white-space:pre-wrap}.users{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px;margin-top:18px}.user{border:1px solid #e2e8f0;border-radius:18px;background:#fff;padding:16px}.uname{font-size:18px;font-weight:950;margin-bottom:8px}.badge{display:inline-block;padding:5px 9px;border-radius:999px;background:#dcfce7;color:#166534;font-size:12px;font-weight:950;margin-right:6px}.badge.off{background:#fee2e2;color:#991b1b}.loader{display:none;position:fixed;inset:0;background:rgba(2,6,23,.55);backdrop-filter:blur(7px);z-index:40;place-items:center}.loader.show{display:grid}.loaderCard{background:#fff;color:#0f172a;border-radius:24px;padding:24px 28px;min-width:320px;text-align:center;box-shadow:var(--shadow)}.spin{width:42px;height:42px;border:4px solid #dbeafe;border-top-color:var(--blue);border-radius:50%;margin:0 auto 12px;animation:spin .85s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}.toast{display:none;position:fixed;right:22px;bottom:22px;background:rgba(15,23,42,.95);border:1px solid var(--border);color:white;min-width:280px;max-width:460px;border-radius:18px;padding:16px 18px;box-shadow:var(--shadow);z-index:50}.toast.show{display:block}@media(max-width:900px){header{flex-direction:column;align-items:flex-start}.status{width:100%}.grid,.grid2{grid-template-columns:1fr}.metrics{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="shell">
<header><div class="brand"><div class="logo"></div><div><h1>Iris Recognition System</h1><div class="sub">Two-eye enrollment • TensorRT FP16 inference • Local web dashboard</div></div></div><div class="status"><div class="line"><span>Status</span><b id="systemStatus">Starting...</b></div><div class="line"><span>Users</span><b id="userCount">0</b></div><div class="line"><span>Server</span><b id="serverIp">Jetson</b></div></div></header>
<div class="tabs"><button class="tab active" data-tab="recognize">Recognition</button><button class="tab" data-tab="enroll">Enrollment / Training</button><button class="tab" data-tab="database">Database</button><button class="tab" data-tab="camera">Camera Later</button></div>

<section id="recognize" class="panel active"><div class="grid"><div class="card"><h2>Recognition</h2><p>Select or drag one iris image. Preview loads automatically; recognition can auto-run after selection.</p><label id="queryDrop" class="drop" for="queryInput"><div class="hint"><div class="hintIcon">◎</div><strong>Drop iris image here or click to choose</strong><span>JPG, PNG, BMP, TIFF supported</span></div><img id="queryPreview"></label><input id="queryInput" type="file" accept="image/*"><label class="check"><input id="autoRecognize" type="checkbox"> Auto-run recognition after choosing image</label><div class="row"><button class="btn green" id="recognizeBtn">Run Recognition</button><button class="btn gray" id="clearQueryBtn">Clear</button></div></div><div class="card dark"><h2>Decision</h2><p style="color:#94a3b8">Compared against both stored eyes of every enrolled user.</p><div class="result"><div id="decisionText" class="decision idle">READY</div><div id="decisionSub" style="color:#cbd5e1">Choose an image to begin.</div><div class="metrics"><div class="metric"><div class="lab">Identity</div><div class="val" id="identityMetric">-</div></div><div class="metric"><div class="lab">Matched Eye</div><div class="val" id="eyeMetric">-</div></div><div class="metric"><div class="lab">Similarity</div><div class="val" id="scoreMetric">-</div></div><div class="metric"><div class="lab">Time</div><div class="val" id="timeMetric">-</div></div></div></div><div style="margin-top:18px;font-weight:950">Threshold</div><div class="slider"><input id="thresholdRange" type="range" min="0.10" max="0.95" step="0.01" value="0.50"><input id="thresholdText" class="input" type="text" value="0.50"></div><div class="log" id="scoreLog">Top matches will appear here.</div></div></div></section>

<section id="enroll" class="panel"><div class="grid"><div class="card"><h2>Enrollment / Training Database</h2><p>This is the demo's training/enrollment step: store a user in the local iris database using left and right eye images.</p><div style="margin-top:18px"><label style="font-weight:950;display:block;margin-bottom:8px">User name</label><input id="enrollName" class="input" type="text" placeholder="Example: Dat"></div><div class="grid2" style="margin-top:18px"><div><h3>Left Eye</h3><label id="leftDrop" class="drop" for="leftInput" style="min-height:230px"><div class="hint"><div class="hintIcon">L</div><strong>Choose left eye</strong><span>Auto preview after selection</span></div><img id="leftPreview"></label><input id="leftInput" type="file" accept="image/*"></div><div><h3>Right Eye</h3><label id="rightDrop" class="drop" for="rightInput" style="min-height:230px"><div class="hint"><div class="hintIcon">R</div><strong>Choose right eye</strong><span>Auto preview after selection</span></div><img id="rightPreview"></label><input id="rightInput" type="file" accept="image/*"></div></div><div class="row"><button class="btn green" id="registerBtn">Enroll User</button><button class="btn gray" id="clearEnrollBtn">Clear</button></div></div><div class="card dark"><h2>Enrollment Result</h2><p style="color:#94a3b8">The system extracts one embedding from each eye and stores both under one identity.</p><div class="result"><div id="enrollDecision" class="decision idle">WAITING</div><div id="enrollSub" style="color:#cbd5e1">Enter a name and choose both eyes.</div></div><div class="metrics"><div class="metric"><div class="lab">Left Eye</div><div class="val" id="leftTimeMetric">-</div></div><div class="metric"><div class="lab">Right Eye</div><div class="val" id="rightTimeMetric">-</div></div><div class="metric"><div class="lab">Total Time</div><div class="val" id="enrollTimeMetric">-</div></div><div class="metric"><div class="lab">Database</div><div class="val" id="enrollDbMetric">-</div></div></div><div class="log">Real flow later:\n1. Capture face\n2. Detect both eyes\n3. Crop iris regions\n4. Extract embeddings\n5. Store both eyes under one identity</div></div></div></section>

<section id="database" class="panel"><div class="grid" style="grid-template-columns:1fr"><div class="card"><div class="row" style="justify-content:space-between;margin-top:0"><div><h2>Database</h2><p>Local JSON enrollment database stored on the Jetson.</p></div><div class="row" style="margin-top:0"><button class="btn" id="refreshDbBtn">Refresh</button><button class="btn red" id="deleteDbBtn">Delete All</button></div></div><div id="usersGrid" class="users"></div></div></div></section>
<section id="camera" class="panel"><div class="grid" style="grid-template-columns:1fr"><div class="card"><h2>Camera Integration Later</h2><p>For tomorrow, image upload mode is cleaner and more stable. Camera capture can be plugged into this same web interface later.</p><div class="log">Next camera upgrade:\n- IMX219 capture endpoint\n- Face image capture\n- Left/right eye detector\n- Iris crop and recognition\n- Keep this same website UI</div></div></div></section>
</div>
<div id="loader" class="loader"><div class="loaderCard"><div class="spin"></div><div id="loaderText" style="font-weight:950">Processing...</div><div style="color:#64748b;margin-top:5px">Jetson is running segmentation and TensorRT inference.</div></div></div><div id="toast" class="toast"></div>
<script>
const state={queryFile:null,leftFile:null,rightFile:null,busy:false}; const $=id=>document.getElementById(id);
function toast(msg){const t=$('toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),3600)}
function loader(msg,on){$('loaderText').textContent=msg||'Processing...';$('loader').classList.toggle('show',!!on)}
function setDecision(cls,main,sub){$('decisionText').className='decision '+cls;$('decisionText').textContent=main;$('decisionSub').textContent=sub}
function setEnroll(cls,main,sub){$('enrollDecision').className='decision '+cls;$('enrollDecision').textContent=main;$('enrollSub').textContent=sub}
function esc(s){return String(s).replace(/[&<>\"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]))}
function tab(id){document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active',b.dataset.tab===id));document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id===id));if(id==='database') users()}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>tab(b.dataset.tab));
async function health(){try{const r=await fetch('/api/health');const d=await r.json();$('systemStatus').textContent=d.ok?'Online':'Error';$('userCount').textContent=d.db_count;$('serverIp').textContent=d.ip+':8000'}catch(e){$('systemStatus').textContent='Offline'}}
async function users(){const g=$('usersGrid');g.innerHTML='<p>Loading...</p>';try{const r=await fetch('/api/users');const d=await r.json();$('userCount').textContent=d.count;if(!d.users.length){g.innerHTML='<p>No enrolled users yet.</p>';return}g.innerHTML='';d.users.forEach(u=>{const div=document.createElement('div');div.className='user';div.innerHTML=`<div class="uname">${esc(u.name)}</div><span class="badge ${u.left?'':'off'}">Left ${u.left?'OK':'Missing'}</span><span class="badge ${u.right?'':'off'}">Right ${u.right?'OK':'Missing'}</span><div style="color:#64748b;margin-top:10px;font-size:13px">Created: ${esc(u.created_at||'unknown')}</div>`;g.appendChild(div)})}catch(e){g.innerHTML='<p>Could not load database.</p>'}}
function setupDrop(dropId,inputId,imgId,key,after){const drop=$(dropId),input=$(inputId),img=$(imgId);function handle(f){if(!f)return;state[key]=f;img.src=URL.createObjectURL(f);drop.classList.add('has');if(after)after(f)}drop.ondragover=e=>{e.preventDefault();drop.classList.add('drag')};drop.ondragleave=()=>drop.classList.remove('drag');drop.ondrop=e=>{e.preventDefault();drop.classList.remove('drag');handle(e.dataTransfer.files[0])};input.onchange=()=>handle(input.files[0])}
setupDrop('queryDrop','queryInput','queryPreview','queryFile',()=>{setDecision('idle','LOADED','Preview ready.');if($('autoRecognize').checked) recognize()});
setupDrop('leftDrop','leftInput','leftPreview','leftFile',()=>setEnroll('idle','LEFT READY','Left eye loaded.'));
setupDrop('rightDrop','rightInput','rightPreview','rightFile',()=>setEnroll('idle','RIGHT READY','Right eye loaded.'));
$('thresholdRange').oninput=()=>$('thresholdText').value=$('thresholdRange').value;$('thresholdText').onchange=()=>{let v=parseFloat($('thresholdText').value);if(isNaN(v))v=.5;v=Math.max(.1,Math.min(.95,v));$('thresholdText').value=v.toFixed(2);$('thresholdRange').value=v.toFixed(2)};
async function recognize(){if(state.busy){toast('Still processing previous image...');return}if(!state.queryFile){toast('Choose an iris image first.');return}state.busy=true;const fd=new FormData();fd.append('image',state.queryFile);fd.append('threshold',$('thresholdText').value||'0.50');loader('Running recognition...',true);try{const r=await fetch('/api/recognize',{method:'POST',body:fd});const d=await r.json();if(!d.ok){setDecision('nomatch','ERROR',d.error||'Recognition failed');toast(d.error||'Recognition failed');return}$('timeMetric').textContent=d.elapsed_ms+' ms';$('scoreMetric').textContent=(d.score===null||d.score===undefined)?'-':Number(d.score).toFixed(6);$('eyeMetric').textContent=d.eye||'-';if(d.matched){$('identityMetric').textContent=d.name||'-';setDecision('match','MATCH FOUND','Identity: '+d.name)}else{$('identityMetric').textContent='Unknown';setDecision('nomatch','NO MATCH','Best score below threshold.')}if(d.top_scores&&d.top_scores.length){$('scoreLog').textContent=d.top_scores.map((s,i)=>`${i+1}. ${s.user} / ${s.eye} : ${Number(s.score).toFixed(6)}`).join('\n')}else{$('scoreLog').textContent='No users in database.'}}catch(e){setDecision('nomatch','ERROR',String(e));toast('Request failed: '+e)}finally{state.busy=false;loader('',false);health()}}
async function enroll(){if(state.busy){toast('Still processing previous request...');return}state.busy=true;const name=$('enrollName').value.trim();if(!name){state.busy=false;toast('Enter user name.');return}if(!state.leftFile||!state.rightFile){state.busy=false;toast('Choose both left and right eye images.');return}const fd=new FormData();fd.append('name',name);fd.append('left',state.leftFile);fd.append('right',state.rightFile);loader('Enrolling both eyes...',true);try{const r=await fetch('/api/register',{method:'POST',body:fd});const d=await r.json();if(!d.ok){setEnroll('nomatch','ERROR',d.error||'Enrollment failed');toast(d.error||'Enrollment failed');return}setEnroll('match','ENROLLED','User added: '+d.name);$('leftTimeMetric').textContent=d.left_ms+' ms';$('rightTimeMetric').textContent=d.right_ms+' ms';$('enrollTimeMetric').textContent=d.elapsed_ms+' ms';$('enrollDbMetric').textContent=d.db.count+' users';toast('Registered both eyes for '+d.name);health();users()}catch(e){setEnroll('nomatch','ERROR',String(e));toast('Request failed: '+e)}finally{state.busy=false;loader('',false)}}
function clearQ(){state.queryFile=null;$('queryInput').value='';$('queryPreview').src='';$('queryDrop').classList.remove('has');$('identityMetric').textContent='-';$('eyeMetric').textContent='-';$('scoreMetric').textContent='-';$('timeMetric').textContent='-';$('scoreLog').textContent='Top matches will appear here.';setDecision('idle','READY','Choose an image to begin.')}
function clearE(){state.leftFile=null;state.rightFile=null;$('leftInput').value='';$('rightInput').value='';$('leftPreview').src='';$('rightPreview').src='';$('leftDrop').classList.remove('has');$('rightDrop').classList.remove('has');$('enrollName').value='';$('leftTimeMetric').textContent='-';$('rightTimeMetric').textContent='-';$('enrollTimeMetric').textContent='-';$('enrollDbMetric').textContent='-';setEnroll('idle','WAITING','Enter a name and choose both eyes.')}
async function delAll(){if(!confirm('Delete all enrolled users?'))return;loader('Clearing database...',true);try{const r=await fetch('/api/delete_all',{method:'POST'});const d=await r.json();toast(d.message||'Database cleared');health();users()}catch(e){toast('Delete failed: '+e)}finally{loader('',false)}}
$('recognizeBtn').onclick=recognize;$('clearQueryBtn').onclick=clearQ;$('registerBtn').onclick=enroll;$('clearEnrollBtn').onclick=clearE;$('refreshDbBtn').onclick=users;$('deleteDbBtn').onclick=delAll;health();users();setInterval(health,7000);
</script>
</body></html>
'''


def main():
    global ARGS, CFG, INFER
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="TensorRT engine path, e.g. iris_model_fp16.engine")
    parser.add_argument("--meta", default="iris_model.metadata.json", help="metadata json from export script")
    parser.add_argument("--db", default=DEFAULT_DB, help="local JSON enrollment database")
    parser.add_argument("--threshold", type=float, default=0.50, help="default cosine threshold")
    parser.add_argument("--host", default="0.0.0.0", help="host address")
    parser.add_argument("--port", type=int, default=8000, help="web server port")
    ARGS = parser.parse_args()

    print("")
    print("=" * 72)
    print(APP_TITLE)
    print("=" * 72)
    print("Loading metadata:", ARGS.meta)
    CFG = core.load_meta(ARGS.meta)
    print("Loading TensorRT engine:", ARGS.engine)
    INFER = core.TRTInfer(ARGS.engine, CFG)
    ip = get_local_ip()
    print("")
    print("Server ready.")
    print("Open on Jetson:        http://127.0.0.1:%d" % ARGS.port)
    print("Open on local network: http://%s:%d" % (ip, ARGS.port))
    print("Database:", ARGS.db)
    print("=" * 72)
    print("")
    app.run(host=ARGS.host, port=ARGS.port, threaded=False, debug=False)


if __name__ == "__main__":
    main()

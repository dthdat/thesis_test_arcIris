#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSON database helpers for the Iris Recognition System web app.

Python 3.6 compatible.
"""

import json
import os
import time

import numpy as np


DB_VERSION = 2
MIN_EMBEDDING_NORM = 1e-6


def now_text():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def empty_db():
    return {"version": DB_VERSION, "users": {}}


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)


def load_db(path):
    if not os.path.exists(path):
        return empty_db()

    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except Exception:
        return empty_db()

    if isinstance(raw, dict) and isinstance(raw.get("users"), dict):
        if "version" not in raw:
            raw["version"] = DB_VERSION
        return raw

    # Migrate old format: {"Dat": [embedding...]}
    db = empty_db()
    if isinstance(raw, dict):
        for name, emb in raw.items():
            if isinstance(emb, list):
                db["users"][str(name)] = {
                    "created_at": "legacy",
                    "left": emb,
                    "right": None,
                    "notes": "Migrated from old single-eye DB"
                }
    return db


def save_db(path, db):
    ensure_parent_dir(path)
    with open(path, "w") as f:
        json.dump(db, f, indent=2)


def reset_db(path):
    save_db(path, empty_db())


def normalize_embedding(embedding):
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise RuntimeError("Embedding is empty")
    if not np.all(np.isfinite(arr)):
        raise RuntimeError("Embedding contains invalid numbers")

    norm = float(np.linalg.norm(arr))
    if (not np.isfinite(norm)) or norm < MIN_EMBEDDING_NORM:
        raise RuntimeError("Embedding norm is zero or invalid")
    return arr / norm


def cosine(a, b):
    aa = normalize_embedding(a)
    bb = normalize_embedding(b)
    return float(np.dot(aa, bb))


def db_summary(path):
    db = load_db(path)
    users = sorted(list(db.get("users", {}).keys()))
    return {"count": len(users), "users": users}


def list_users(path):
    db = load_db(path)
    users = []
    for name, rec in sorted(db.get("users", {}).items()):
        if not isinstance(rec, dict):
            continue
        users.append({
            "name": name,
            "created_at": rec.get("created_at", "unknown"),
            "left": rec.get("left") is not None,
            "right": rec.get("right") is not None,
            "notes": rec.get("notes", "")
        })
    return users


def register_both(path, name, left_emb, right_emb):
    clean_name = (name or "").strip()
    if not clean_name:
        raise RuntimeError("Name is required")

    left = normalize_embedding(left_emb).tolist()
    right = normalize_embedding(right_emb).tolist()

    db = load_db(path)
    db["users"][clean_name] = {
        "created_at": now_text(),
        "left": left,
        "right": right,
        "notes": "Two-eye enrollment from web UI"
    }
    save_db(path, db)
    return db_summary(path)


def recognize(path, embedding, threshold, top_k):
    query = normalize_embedding(embedding)
    db = load_db(path)
    users = db.get("users", {})
    if not users:
        return None, None, None, []

    best_name = None
    best_eye = None
    best_score = -999.0
    top = []

    for name, rec in users.items():
        if not isinstance(rec, dict):
            continue
        for eye in ["left", "right"]:
            ref = rec.get(eye)
            if ref is None:
                continue
            try:
                score = cosine(query, ref)
                bad = False
            except Exception:
                score = 0.0
                bad = True

            top.append({
                "user": name,
                "eye": eye,
                "score": score,
                "bad_embedding": bad
            })
            if (not bad) and score > best_score:
                best_name = name
                best_eye = eye
                best_score = score

    top.sort(key=lambda x: x["score"], reverse=True)
    if best_score <= -998.0:
        return None, None, 0.0, top[:top_k]
    if best_score >= threshold:
        return best_name, best_eye, best_score, top[:top_k]
    return None, best_eye, best_score, top[:top_k]

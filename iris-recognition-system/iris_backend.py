#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jetson backend wrapper for iris preprocessing and TensorRT inference.

This module intentionally delegates the model-specific work to the existing
~/iris_app_py36.py file used by the thesis demo.

Python 3.6 compatible.
"""

import json
import os
import sys
import threading

import numpy as np


HOME = os.path.expanduser("~")
if HOME not in sys.path:
    sys.path.insert(0, HOME)

try:
    import iris_app_py36 as core
except Exception as import_error:
    core = None
    CORE_IMPORT_ERROR = import_error
else:
    CORE_IMPORT_ERROR = None


MIN_EMBEDDING_NORM = 1e-6


class IrisBackend(object):
    def __init__(self, engine_path, meta_path):
        if core is None:
            raise RuntimeError(
                "Could not import ~/iris_app_py36.py: " + str(CORE_IMPORT_ERROR)
            )
        self.engine_path = engine_path
        self.meta_path = meta_path
        self.cfg = None
        self.infer = None
        self.lock = threading.Lock()

    def load(self):
        self.cfg = core.load_meta(self.meta_path)
        self.infer = core.TRTInfer(self.engine_path, self.cfg)

    def make_embedding(self, image_path):
        if self.cfg is None or self.infer is None:
            raise RuntimeError("Backend is not loaded")

        # TensorRT/PyCUDA context access is serialized for Jetson stability.
        with self.lock:
            polar, meta = core.preprocess_image(image_path, self.cfg)
            if polar is None:
                reason = "unknown"
                if isinstance(meta, dict):
                    reason = meta.get("reason", "unknown")
                raise RuntimeError("Iris segmentation failed: " + str(reason))
            emb = self.infer(polar)

        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if emb.size == 0:
            raise RuntimeError("TensorRT returned an empty embedding")
        if not np.all(np.isfinite(emb)):
            raise RuntimeError("TensorRT returned an embedding with invalid numbers")

        emb_norm = float(np.linalg.norm(emb))
        if (not np.isfinite(emb_norm)) or emb_norm < MIN_EMBEDDING_NORM:
            raise RuntimeError(
                "TensorRT returned a zero/invalid embedding. Restart the server, "
                "reset the old database if needed, and re-enroll users."
            )

        if isinstance(meta, dict):
            meta["embedding_norm"] = emb_norm
            meta["embedding_dim"] = int(emb.shape[0])
        return emb, meta


def meta_to_json(meta):
    if not isinstance(meta, dict):
        return {}
    out = {}
    for key, value in meta.items():
        if isinstance(value, (tuple, list)):
            vals = []
            for item in value:
                try:
                    vals.append(float(item))
                except Exception:
                    vals.append(str(item))
            out[key] = vals
        elif isinstance(value, np.floating):
            out[key] = float(value)
        elif isinstance(value, np.integer):
            out[key] = int(value)
        else:
            try:
                json.dumps(value)
                out[key] = value
            except Exception:
                out[key] = str(value)
    return out

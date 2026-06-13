# Iris Recognition System

Web dashboard for the Jetson Nano thesis demo. The app keeps the working TensorRT backend in `~/iris_app_py36.py` and provides a cleaner browser UI for enrollment, recognition, and database checks.

## Jetson Environment

- Ubuntu 18.04.6 LTS
- Python 3.6.9
- JetPack 4.6.x
- Existing backend: `~/iris_app_py36.py`
- Existing TensorRT engine: `~/Documents/thesis/iris_model_fp16.engine`

## Install

Prefer Jetson apt packages:

```bash
sudo apt update
sudo apt install python3-flask python3-pil python3-pil.imagetk python3-opencv python3-scipy -y
```

Only use `requirements_jetson.txt` as a pip fallback for Flask on Python 3.6.

## Run

```bash
cd ~/Documents/thesis/iris-recognition-system
bash scripts/run_server.sh
```

The server binds to `0.0.0.0:8000` by default and prints the LAN URL on startup.
If `../iris_users_web.json` already exists, the run script uses it so existing enrollments are preserved. Otherwise it uses `data/iris_users_web.json`.

To check the Jetson IP:

```bash
hostname -I
```

Open from a laptop or phone on the same network:

```text
http://JETSON_IP:8000
```

For this deployment target:

```text
http://192.168.2.49:8000
```

## Demo Flow

1. Open Enrollment / Training.
2. Enter the user name.
3. Upload the left-eye image.
4. Upload the right-eye image.
5. Click Enroll User.
6. Open Recognition.
7. Upload a query eye image.
8. Click Run Recognition.

Auto-run recognition is off by default. The default threshold is `0.30`.

## API

- `GET /`
- `GET /api/health`
- `GET /api/users`
- `POST /api/register`
- `POST /api/recognize`
- `POST /api/delete_all`

## Safety

- `scripts/reset_db.sh` resets only `data/iris_users_web.json`.
- The app does not modify model, ONNX, TensorRT engine, notebook, or training files.
- Flask runs with `threaded=False`.
- TensorRT inference is protected by a server-side lock.
- `/api/health` does not touch TensorRT.

## Troubleshooting

- All scores are `0.0000`: reset the DB and re-enroll users with the fixed server.
- CUDA invalid resource handle: stop duplicate servers, keep `threaded=False`, and restart the app.
- PyCUDA import error: check CUDA/PyCUDA paths on the Jetson.
- Segmentation failed: choose a clearer iris image.
- Engine not found: set `IRIS_ENGINE=/path/to/iris_model_fp16.engine`.
- Metadata not found: set `IRIS_META=/path/to/metadata.json`.

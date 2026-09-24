"""Model registry. Weights are referenced by URL and SHA-256 and never committed.

detector_weights.py resolves the appearance-channel detector from training/results/hf_model.json
(or YOLO_MODEL_URL + YOLO_MODEL_SHA256), downloads and verifies it, and opens it with onnxruntime.
"""

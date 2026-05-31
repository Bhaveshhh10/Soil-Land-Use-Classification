"""Resolve trained model artefacts from local, Colab, or env paths."""

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ART_DIR = os.path.join(BASE_DIR, "artefacts")


def _candidate_dirs():
    env_dir = os.environ.get("ARTEFACTS_DIR", "").strip()
    return [
        env_dir or None,
        os.path.join(BASE_DIR, "artefacts"),
        os.path.join(BASE_DIR, "..", "artefacts"),
        os.path.join(BASE_DIR, "..", "streamlit_app", "artefacts"),
        "/content/streamlit_app/artefacts",
        "/content/artefacts",
        "/kaggle/working/artefacts",
    ]


def resolve_artefacts_dir():
    """Prefer a directory that actually contains artefacts.json."""
    for path in _candidate_dirs():
        if not path:
            continue
        json_path = os.path.join(path, "artefacts.json")
        if os.path.isfile(json_path):
            return os.path.abspath(path)
    for path in _candidate_dirs():
        if path and os.path.isdir(path):
            return os.path.abspath(path)
    return DEFAULT_ART_DIR


def artefacts_json_path():
    return os.path.join(resolve_artefacts_dir(), "artefacts.json")


def artefact_paths():
    art_dir = resolve_artefacts_dir()
    return {
        "dir": art_dir,
        "model": os.path.join(art_dir, "se_mobilenetv2_eurosat.h5"),
        "json": os.path.join(art_dir, "artefacts.json"),
        "knn": os.path.join(art_dir, "knn_fallback.pkl"),
        "scaler": os.path.join(art_dir, "emb_scaler.pkl"),
        "x_test": os.path.join(art_dir, "X_test.npy"),
        "y_test": os.path.join(art_dir, "y_test.npy"),
    }


def bootstrap_artefacts_json():
    """Create artefacts.json if the folder exists but metrics file does not."""
    art_dir = resolve_artefacts_dir()
    os.makedirs(art_dir, exist_ok=True)
    json_path = os.path.join(art_dir, "artefacts.json")
    if os.path.isfile(json_path):
        return json_path

    nb_path = os.path.join(BASE_DIR, "..", "notebook28be1f76d9.ipynb")
    extract_script = os.path.join(BASE_DIR, "extract_notebook_metrics.py")
    if os.path.isfile(nb_path) and os.path.isfile(extract_script):
        subprocess.run([sys.executable, extract_script], cwd=BASE_DIR, check=False)
        if os.path.isfile(json_path):
            return json_path

    gen_script = os.path.join(BASE_DIR, "generate_demo_artefacts.py")
    if os.path.isfile(gen_script):
        subprocess.run([sys.executable, gen_script], cwd=BASE_DIR, check=False)
    if os.path.isfile(json_path):
        return json_path

    return json_path if os.path.isfile(json_path) else None


def ensure_artefacts():
    """Guarantee artefacts.json exists; return resolved paths."""
    bootstrap_artefacts_json()
    return artefact_paths()

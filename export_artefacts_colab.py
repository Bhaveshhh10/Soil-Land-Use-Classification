"""
Run this at the end of your Colab/Kaggle notebook after training (cells 1–13).

It writes the same files as train.py so the Streamlit dashboard can load them.

Usage in Colab:
    %run export_artefacts_colab.py

Or paste the export_artefacts() call into your last notebook cell.
"""

import json
import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def export_artefacts(
    out_dir="/content/streamlit_app/artefacts",
    se_model=None,
    knn_fallback=None,
    emb_scaler=None,
    X_test=None,
    y_test=None,
    routed_preds=None,
    pred_sources=None,
    se_confidences=None,
    ml_results=None,
    van_acc=None,
    van_f1=None,
    se_acc=None,
    se_f1=None,
    HISTORY=None,
    difficulty=None,
    tier_easy=None,
    tier_med=None,
    clustering=None,
    classes=None,
):
    """Export all dashboard artefacts. Pass variables from the training notebook."""
    os.makedirs(out_dir, exist_ok=True)

    if se_model is not None:
        se_model.save(os.path.join(out_dir, "se_mobilenetv2_eurosat.h5"))

    if knn_fallback is not None:
        with open(os.path.join(out_dir, "knn_fallback.pkl"), "wb") as f:
            pickle.dump(knn_fallback, f)
    if emb_scaler is not None:
        with open(os.path.join(out_dir, "emb_scaler.pkl"), "wb") as f:
            pickle.dump(emb_scaler, f)

    if X_test is not None:
        np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    if y_test is not None:
        np.save(os.path.join(out_dir, "y_test.npy"), y_test)

    if se_acc is None and se_model is not None and X_test is not None and y_test is not None:
        se_pred = np.argmax(se_model.predict(X_test, verbose=0), axis=1)
        se_acc = accuracy_score(y_test, se_pred) * 100
        se_f1 = f1_score(y_test, se_pred, average="macro")

    if routed_preds is not None and y_test is not None:
        routed_acc = accuracy_score(y_test, routed_preds) * 100
        routed_f1 = f1_score(y_test, routed_preds, average="macro")
    else:
        routed_acc = se_acc
        routed_f1 = se_f1

    n_test = len(y_test) if y_test is not None else 0
    n_knn = int((pred_sources == "knn_fallback").sum()) if pred_sources is not None else 0

    ml_clean = {}
    if ml_results:
        for name, vals in ml_results.items():
            ml_clean[name] = {"acc": vals.get("acc", 0), "f1": vals.get("f1", 0)}

    artefacts = {
        "classes": list(classes) if classes is not None else [],
        "num_classes": len(classes) if classes is not None else 10,
        "img_size": 64,
        "se_acc": float(se_acc or 0),
        "se_f1": float(se_f1 or 0),
        "van_acc": float(van_acc or 0),
        "van_f1": float(van_f1 or 0),
        "routed_acc": float(routed_acc or 0),
        "routed_f1": float(routed_f1 or 0),
        "n_se_direct": n_test - n_knn,
        "n_knn_routed": n_knn,
        "n_test": n_test,
        "avg_se_conf": float(np.mean(se_confidences)) if se_confidences is not None else 0.0,
        "ml_results": ml_clean,
        "clustering": clustering or {},
        "history": {
            "phase1_val_acc": HISTORY["phase1"]["val_accuracy"] if HISTORY else [],
            "phase2_val_acc": HISTORY["phase2"]["val_accuracy"] if HISTORY else [],
            "phase3_val_acc": HISTORY["phase3"]["val_accuracy"] if HISTORY else [],
        },
        "difficulty": difficulty.tolist() if hasattr(difficulty, "tolist") else (difficulty or []),
        "tier_easy_max": float(difficulty[tier_easy].max()) if difficulty is not None and tier_easy is not None else 0.33,
        "tier_med_max": float(difficulty[tier_med].max()) if difficulty is not None and tier_med is not None else 0.66,
        "confidence_threshold": 0.60,
        "source": "colab_export",
    }

    json_path = os.path.join(out_dir, "artefacts.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artefacts, f, indent=2)

    print(f"Exported artefacts to {out_dir}")
    return json_path


if __name__ == "__main__":
    print(
        "Import export_artefacts() in your notebook and call it with training variables.\n"
        "See docstring for required arguments."
    )

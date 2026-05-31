"""
DEPRECATED — use extract_notebook_metrics.py for real Kaggle notebook results.

    cd streamlit_app
    python extract_notebook_metrics.py

Only use this if the notebook file is not available.
"""

import json
import os
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "artefacts")
os.makedirs(OUT_DIR, exist_ok=True)

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

rng = np.random.default_rng(42)


def _phase_curve(start, end, epochs, noise=0.008):
    t = np.linspace(0, 1, epochs)
    curve = start + (end - start) * (1 - np.exp(-3.5 * t))
    curve += rng.normal(0, noise, epochs)
    return np.clip(curve, 0, 1).tolist()


def main():
    n_test = 5400
    n_knn = 812

    artefacts = {
        "classes": EUROSAT_CLASSES,
        "num_classes": 10,
        "img_size": 64,
        "se_acc": 98.42,
        "se_f1": 0.9831,
        "van_acc": 96.71,
        "van_f1": 0.9668,
        "routed_acc": 99.08,
        "routed_f1": 0.9902,
        "n_se_direct": n_test - n_knn,
        "n_knn_routed": n_knn,
        "n_test": n_test,
        "avg_se_conf": 0.847,
        "ml_results": {
            "Logistic Regression": {"acc": 74.2, "f1": 0.731},
            "KNN (k=5)": {"acc": 86.5, "f1": 0.862},
            "Decision Tree": {"acc": 79.8, "f1": 0.791},
            "Random Forest": {"acc": 93.1, "f1": 0.928},
        },
        "clustering": {
            "km_ari": 0.621,
            "km_nmi": 0.708,
            "km_sil": 0.384,
            "kmed_ari": 0.587,
            "kmed_sil": 0.412,
        },
        "history": {
            "phase1_val_acc": _phase_curve(0.52, 0.89, 12),
            "phase2_val_acc": _phase_curve(0.88, 0.95, 10),
            "phase3_val_acc": _phase_curve(0.94, 0.987, 18),
        },
        "difficulty": rng.beta(2, 5, size=1200).tolist(),
        "tier_easy_max": 0.33,
        "tier_med_max": 0.66,
        "confidence_threshold": 0.60,
        "source": "demo_bootstrap",
    }

    path = os.path.join(OUT_DIR, "artefacts.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artefacts, f, indent=2)

    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

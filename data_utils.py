"""
data_utils.py — Dataset loading, preprocessing, and EDA utilities.
Exact port from notebook28be1f76d9.ipynb cells 2–4.
"""

import os
import glob
import warnings
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

IMG_SIZE = 64
RANDOM_STATE = 42

# ─── EuroSAT class list (canonical order) ─────────────────────────────────────
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Accent colours for class cards (professional UI)
CLASS_COLORS = {
    "AnnualCrop": "#c9a227",
    "Forest": "#2d6a4f",
    "HerbaceousVegetation": "#52b788",
    "Highway": "#6c757d",
    "Industrial": "#495057",
    "Pasture": "#a7c957",
    "PermanentCrop": "#9b2226",
    "Residential": "#4a4e69",
    "River": "#0077b6",
    "SeaLake": "#023e8a",
}

# Legacy alias — avoid emojis in UI
CLASS_EMOJI = {k: "" for k in CLASS_COLORS}


def find_dataset_dir(search_roots=None):
    """
    Auto-detect EuroSAT dataset directory.
    Checks common Kaggle/Colab paths and one level deeper.
    """
    if search_roots is None:
        search_roots = [
            "/kaggle/input/datasets/waseemalastal/eurosat-rgb-dataset",
            "/kaggle/input/eurosat-dataset",
            "/kaggle/input/eurosat",
            "/content/EuroSAT",
            "/content/eurosat",
            "data/EuroSAT",
            "data/eurosat",
            os.path.join(os.path.dirname(__file__), "..", "data", "EuroSAT"),
        ]

    for root in search_roots:
        if not os.path.exists(root):
            continue
        candidates = glob.glob(os.path.join(root, "**"), recursive=False)
        class_dirs = [c for c in candidates if os.path.isdir(c)]
        if len(class_dirs) >= 5:
            return root
        # Check one level deeper
        for sub in class_dirs:
            inner = [
                c
                for c in glob.glob(os.path.join(sub, "**"), recursive=False)
                if os.path.isdir(c)
            ]
            if len(inner) >= 5:
                return sub
    return None


def load_images(image_dir, classes, img_size=IMG_SIZE, max_per_class=None):
    """
    Load all images from the EuroSAT directory.
    Returns (X, y) as float32 numpy arrays, normalised to [0,1].
    """
    X, y = [], []
    for label, cls in enumerate(classes):
        cls_path = os.path.join(image_dir, cls)
        img_files = (
            glob.glob(os.path.join(cls_path, "*.jpg"))
            + glob.glob(os.path.join(cls_path, "*.png"))
            + glob.glob(os.path.join(cls_path, "*.tif"))
        )
        if max_per_class:
            img_files = img_files[:max_per_class]

        for img_path in img_files:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32) / 255.0
            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)


def split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Stratified 80/20 train-test split."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def preprocess_single_image(pil_img, img_size=IMG_SIZE):
    """
    Preprocess a PIL image for inference.
    Returns float32 numpy array of shape (img_size, img_size, 3), normalised to [0,1].
    """
    img_np = np.array(pil_img.convert("RGB").resize((img_size, img_size)))
    return img_np.astype(np.float32) / 255.0


def get_class_counts(image_dir, classes):
    """Return dict of {class_name: image_count}."""
    counts = {}
    for cls in classes:
        cls_path = os.path.join(image_dir, cls)
        files = (
            glob.glob(os.path.join(cls_path, "*.jpg"))
            + glob.glob(os.path.join(cls_path, "*.png"))
            + glob.glob(os.path.join(cls_path, "*.tif"))
        )
        counts[cls] = len(files)
    return counts


def make_sample_grid_figure(image_dir, classes, img_size=IMG_SIZE):
    """
    Create a matplotlib figure showing one sample image per class.
    Returns the figure object.
    """
    cols = 5
    rows = (len(classes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle("EuroSAT Dataset — One Sample per Class", fontsize=14, fontweight="bold", y=1.01)

    for i, cls in enumerate(classes):
        cls_path = os.path.join(image_dir, cls)
        img_files = (
            glob.glob(os.path.join(cls_path, "*.jpg"))
            + glob.glob(os.path.join(cls_path, "*.png"))
            + glob.glob(os.path.join(cls_path, "*.tif"))
        )
        ax = axes.flat[i]
        if img_files:
            img = cv2.cvtColor(cv2.imread(img_files[0]), cv2.COLOR_BGR2RGB)
            ax.imshow(cv2.resize(img, (img_size, img_size)))
        emoji = CLASS_EMOJI.get(cls, "")
        ax.set_title(f"{emoji} {cls}", fontsize=8, fontweight="bold")
        ax.axis("off")

    for ax in axes.flat[len(classes) :]:
        ax.axis("off")

    plt.tight_layout()
    return fig

"""
model.py — SE-MobileNetV2 architecture, Focal Loss, and all ML utilities.
Exact port from notebook28be1f76d9.ipynb.
"""

import os
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 64
BATCH_SIZE = 32
RANDOM_STATE = 42
CONFIDENCE_THRESHOLD = 0.60

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
NUM_CLASSES = len(EUROSAT_CLASSES)

# ─── Focal Loss ───────────────────────────────────────────────────────────────

def focal_loss(gamma: float = 2.0, alpha: float = 0.25, num_classes: int = NUM_CLASSES):
    """
    Focal Loss for multi-class classification.
    L_focal = -alpha * (1 - p_t)^gamma * log(p_t)
    Forces the model to focus on hard, misclassified examples.
    """
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        y_true_1d = tf.squeeze(y_true, axis=-1) if len(y_true.shape) > 1 else y_true
        y_true_oh = tf.one_hot(y_true_1d, depth=num_classes)

        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        ce = -tf.math.log(p_t)
        loss = alpha * focal_weight * ce
        return tf.reduce_mean(loss)

    focal_loss_fn.__name__ = f"focal_loss_g{gamma}_a{alpha}"
    return focal_loss_fn


# ─── SE Block ─────────────────────────────────────────────────────────────────

def se_block(x, ratio: int = 16, name_prefix: str = ""):
    """
    Squeeze-and-Excitation channel attention block.
    Squeeze: GlobalAveragePooling → (1,1,C)
    Excitation: FC(C/r, relu) → FC(C, sigmoid)
    Scale: element-wise multiply with input
    """
    channels = x.shape[-1]

    se = layers.GlobalAveragePooling2D(name=f"{name_prefix}_se_gap")(x)
    se = layers.Reshape((1, 1, channels), name=f"{name_prefix}_se_reshape")(se)
    se = layers.Conv2D(
        max(channels // ratio, 1), (1, 1),
        activation="relu", use_bias=True,
        name=f"{name_prefix}_se_fc1",
    )(se)
    se = layers.Conv2D(
        channels, (1, 1),
        activation="sigmoid", use_bias=True,
        name=f"{name_prefix}_se_fc2",
    )(se)
    return layers.Multiply(name=f"{name_prefix}_se_scale")([x, se])


# ─── SE-MobileNetV2 ───────────────────────────────────────────────────────────

def build_se_mobilenetv2(
    num_classes: int = NUM_CLASSES,
    img_size: int = IMG_SIZE,
    se_ratio: int = 16,
    dropout_rate: float = 0.4,
):
    """
    SE-MobileNetV2 (multi-scale variant).
    SE blocks applied to 4 intermediate feature maps from MobileNetV2 backbone,
    then fused via concatenation.

    Feature scales:
      block_3_expand_relu  : early  (16×16×96)
      block_6_expand_relu  : mid    (8×8×192)
      block_13_expand_relu : deep   (4×4×576)
      out_relu             : final  (2×2×1280)
    """
    inp = keras.Input(shape=(img_size, img_size, 3), name="input_image")
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )

    feature_layer_names = [
        "block_3_expand_relu",
        "block_6_expand_relu",
        "block_13_expand_relu",
        "out_relu",
    ]

    available = {l.name for l in base.layers}
    valid_names = [n for n in feature_layer_names if n in available]
    if not valid_names:
        valid_names = [base.layers[-1].name]

    feat_outputs = [base.get_layer(n).output for n in valid_names]
    feat_model = keras.Model(base.input, feat_outputs, name="backbone_multiscale")
    feat_model.trainable = False  # frozen initially

    feats = feat_model(inp, training=False)
    if not isinstance(feats, list):
        feats = [feats]

    pooled = []
    for i, f in enumerate(feats):
        f_se = se_block(f, ratio=se_ratio, name_prefix=f"scale_{i}")
        f_pool = layers.GlobalAveragePooling2D(name=f"gap_scale_{i}")(f_se)
        pooled.append(f_pool)

    x = layers.Concatenate(name="multiscale_concat")(pooled) if len(pooled) > 1 else pooled[0]
    x = layers.Dense(512, activation="relu", name="head_fc1")(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(dropout_rate, name="head_drop1")(x)
    x = layers.Dense(256, activation="relu", name="head_fc2")(x)
    x = layers.Dropout(dropout_rate * 0.75, name="head_drop2")(x)
    out_cls = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inp, outputs=out_cls, name="SE_MobileNetV2_MultiScale")
    return model, feat_model


# ─── Curriculum Learning ──────────────────────────────────────────────────────

class CurriculumDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence that presents samples in curriculum order.
    The active tier (easy/medium/full) is set externally before each phase.
    Includes online augmentation.
    """

    def __init__(self, X, y, indices, batch_size: int = 32, augment: bool = True):
        self.X = X
        self.y = y
        self.indices = indices.copy()
        self.batch_size = batch_size
        self.augment = augment
        self.aug = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.15,
            fill_mode="nearest",
        )

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        Xb = self.X[batch_idx].copy()
        yb = self.y[batch_idx]
        if self.augment:
            Xb = np.array([self.aug.random_transform(x) for x in Xb])
        return Xb, yb

    def set_tier(self, new_indices):
        self.indices = new_indices.copy()
        np.random.shuffle(self.indices)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def compute_curriculum_tiers(X_train, y_train, random_state: int = RANDOM_STATE):
    """
    Compute per-sample difficulty using a Random Forest proxy on PCA features.
    Returns (difficulty, tier_easy, tier_med, tier_full, X_sc_train, X_sc_test_fn)
    where X_sc_test_fn is a callable that transforms test data.
    """
    X_flat = X_train.reshape(len(X_train), -1)

    pca = PCA(n_components=100, random_state=random_state)
    scaler = StandardScaler()

    X_pca = pca.fit_transform(X_flat)
    X_sc = scaler.fit_transform(X_pca)

    proxy_rf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)
    proxy_rf.fit(X_sc, y_train)
    train_proba = proxy_rf.predict_proba(X_sc)

    max_conf = np.max(train_proba, axis=1)
    difficulty = 1.0 - max_conf

    sorted_idx = np.argsort(difficulty)
    n = len(X_train)
    tier_easy = sorted_idx[: int(0.40 * n)]
    tier_med = sorted_idx[: int(0.70 * n)]
    tier_full = sorted_idx

    return difficulty, tier_easy, tier_med, tier_full, X_sc, pca, scaler


# ─── Agentic Confidence Router ────────────────────────────────────────────────

def build_feature_extractor(se_model):
    """Build a feature extractor from the SE model (embeddings before softmax)."""
    try:
        embedding_layer = se_model.get_layer("head_fc2")
    except ValueError:
        embedding_layer = se_model.get_layer("head_fc1")

    return keras.Model(
        inputs=se_model.input,
        outputs=embedding_layer.output,
        name="SE_FeatureExtractor",
    )


def build_knn_fallback(feature_extractor, X_train, y_train, batch_size: int = 64):
    """Train KNN on SE embeddings for the fallback specialist."""
    emb_train = feature_extractor.predict(X_train, batch_size=batch_size, verbose=0)
    emb_scaler = StandardScaler()
    emb_train_sc = emb_scaler.fit_transform(emb_train)

    knn = KNeighborsClassifier(n_neighbors=7, metric="cosine", n_jobs=-1)
    knn.fit(emb_train_sc, y_train)
    return knn, emb_scaler


def agentic_router(
    images,
    se_model,
    knn_fallback,
    feature_extractor,
    emb_scaler,
    threshold: float = CONFIDENCE_THRESHOLD,
):
    """
    Agentic Confidence Router.
    - confidence >= threshold  →  SE direct prediction
    - confidence <  threshold  →  KNN fallback on SE embeddings

    Returns: (final_preds, sources, se_confidences)
    """
    se_proba = se_model.predict(images, batch_size=64, verbose=0)
    se_preds = np.argmax(se_proba, axis=1)
    se_conf = np.max(se_proba, axis=1)

    low_conf_mask = se_conf < threshold
    n_routed = low_conf_mask.sum()

    final_preds = se_preds.copy()
    sources = np.array(["se_direct"] * len(images), dtype=object)

    if n_routed > 0:
        emb_low = feature_extractor.predict(images[low_conf_mask], batch_size=64, verbose=0)
        emb_low_sc = emb_scaler.transform(emb_low)
        knn_preds = knn_fallback.predict(emb_low_sc)
        final_preds[low_conf_mask] = knn_preds
        sources[low_conf_mask] = "knn_fallback"

    return final_preds, sources, se_conf, se_proba


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────

def make_gradcam(model, img_array, img_size: int = IMG_SIZE):
    """
    Class-discriminative Grad-CAM for SE-MobileNetV2.
    Finds the last Conv2D in the backbone sub-model.
    Returns (cam_resized, overlay, pred_idx, confidence) or None on failure.
    Uses a persistent GradientTape watching the conv output directly.
    """
    import cv2

    # ── find last Conv2D ─────────────────────────────────────────────────────
    target_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            target_layer = layer
            break
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    target_layer = sub
                    break
        if target_layer:
            break

    if target_layer is None:
        return None

    # ── build grad model ─────────────────────────────────────────────────────
    try:
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )
    except Exception:
        grad_model = None
        for sm in [l for l in model.layers if hasattr(l, "layers")]:
            try:
                grad_model = keras.Model(
                    inputs=model.inputs,
                    outputs=[sm.get_layer(target_layer.name).output, model.output],
                )
                break
            except Exception:
                continue
        if grad_model is None:
            return None

    try:
        inp = tf.cast(img_array[np.newaxis], tf.float32)

        # persistent tape so we can call gradient after the context
        with tf.GradientTape(persistent=True) as tape:
            conv_out, preds = grad_model(inp)
            tape.watch(conv_out)
            pred_idx   = int(np.argmax(np.array(preds)[0]))
            confidence = float(np.array(preds)[0, pred_idx])
            loss       = preds[:, pred_idx]

        grads = tape.gradient(loss, conv_out)
        del tape

        if grads is None:
            # fallback: mean activation map
            cam = np.mean(np.array(conv_out)[0], axis=-1)
        else:
            weights = np.mean(np.array(grads), axis=(0, 1, 2))   # (C,)
            cam     = np.dot(np.array(conv_out)[0], weights)      # (H, W)

        cam = np.maximum(cam, 0).astype(np.float32)
        if cam.max() > 0:
            cam /= cam.max()

        cam_resized = cv2.resize(cam, (img_size, img_size))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        overlay = np.clip(0.55 * img_array + 0.45 * heatmap, 0, 1)

        return cam_resized, overlay, pred_idx, confidence

    except Exception:
        return None


# ─── K-Medoids ────────────────────────────────────────────────────────────────

def k_medoids_fit(X, k: int = 5, n_iter: int = 8, seed: int = 42):
    """Simple K-Medoids via alternating assignment and medoid update."""
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(X), k, replace=False)
    for _ in range(n_iter):
        D = pairwise_distances(X, X[idxs])
        labels = np.argmin(D, axis=1)
        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members):
                D_intra = pairwise_distances(X[members]).sum(axis=1)
                idxs[j] = members[np.argmin(D_intra)]
    return labels

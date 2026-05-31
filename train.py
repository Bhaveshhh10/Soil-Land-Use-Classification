"""
train.py — Full 3-phase curriculum training pipeline.
Exact port from notebook28be1f76d9.ipynb cells 7–9.
Run this script once to train and save the model + artefacts.

Usage:
    python train.py --data_dir /path/to/EuroSAT_RGB --out_dir ./artefacts
"""

import os
import argparse
import warnings
import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from data_utils import load_images, split_data, find_dataset_dir, EUROSAT_CLASSES
from model import (
    focal_loss,
    build_se_mobilenetv2,
    CurriculumDataGenerator,
    compute_curriculum_tiers,
    build_feature_extractor,
    build_knn_fallback,
    agentic_router,
    k_medoids_fit,
    IMG_SIZE,
    BATCH_SIZE,
    RANDOM_STATE,
    NUM_CLASSES,
)

warnings.filterwarnings("ignore")
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def get_callbacks_list(monitor="val_accuracy", patience_es=6, patience_lr=3):
    return [
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            patience=patience_lr,
            factor=0.5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def train(data_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("⏳ Loading images...")
    classes = EUROSAT_CLASSES
    X, y = load_images(data_dir, classes, img_size=IMG_SIZE)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"✅ Train: {len(X_train)}  Test: {len(X_test)}  Classes: {NUM_CLASSES}")

    # ── 2. Focal Loss ─────────────────────────────────────────────────────────
    FL = focal_loss(gamma=2.0, alpha=0.25, num_classes=NUM_CLASSES)

    # ── 3. Build SE-MobileNetV2 ───────────────────────────────────────────────
    print("🏗️  Building SE-MobileNetV2...")
    se_model, backbone = build_se_mobilenetv2(
        num_classes=NUM_CLASSES, img_size=IMG_SIZE, se_ratio=16, dropout_rate=0.4
    )
    print(f"✅ Model built: {se_model.count_params():,} params")

    # ── 4. Curriculum Learning setup ──────────────────────────────────────────
    print("⏳ Computing curriculum difficulty scores...")
    difficulty, tier_easy, tier_med, tier_full, X_sc_train, pca, scaler = (
        compute_curriculum_tiers(X_train, y_train, random_state=RANDOM_STATE)
    )

    X_flat_test = X_test.reshape(len(X_test), -1)
    X_pca_test = pca.transform(X_flat_test)
    X_sc_test = scaler.transform(X_pca_test)

    gen_easy = CurriculumDataGenerator(X_train, y_train, tier_easy, BATCH_SIZE)
    gen_med = CurriculumDataGenerator(X_train, y_train, tier_med, BATCH_SIZE)
    gen_full = CurriculumDataGenerator(X_train, y_train, tier_full, BATCH_SIZE)
    val_gen = CurriculumDataGenerator(
        X_test, y_test, np.arange(len(X_test)), BATCH_SIZE, augment=False
    )

    HISTORY = {}

    # ── 5. Phase 1: Warm-up (Easy, frozen) ───────────────────────────────────
    print("\n" + "=" * 60)
    print("📚 Phase 1: Curriculum Warm-up (Easy 40%, backbone frozen)")
    print("=" * 60)
    backbone.trainable = False
    se_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=FL,
        metrics=["accuracy"],
    )
    hist_p1 = se_model.fit(
        gen_easy,
        validation_data=val_gen,
        epochs=12,
        callbacks=get_callbacks_list(patience_es=5, patience_lr=2),
        verbose=1,
    )
    HISTORY["phase1"] = hist_p1.history
    print(f"✅ Phase 1 done | Best val acc: {max(hist_p1.history['val_accuracy']):.4f}")

    # ── 6. Phase 2: Core (Medium, frozen) ────────────────────────────────────
    print("\n" + "=" * 60)
    print("📚 Phase 2: Curriculum Core (Medium 70%, backbone frozen)")
    print("=" * 60)
    hist_p2 = se_model.fit(
        gen_med,
        validation_data=val_gen,
        epochs=15,
        callbacks=get_callbacks_list(patience_es=6, patience_lr=3),
        verbose=1,
    )
    HISTORY["phase2"] = hist_p2.history
    print(f"✅ Phase 2 done | Best val acc: {max(hist_p2.history['val_accuracy']):.4f}")

    # ── 7. Phase 3: Fine-tuning (Full, top-30 unfrozen) ──────────────────────
    print("\n" + "=" * 60)
    print("🔧 Phase 3: Fine-tuning (Full 100%, top-30 layers unfrozen)")
    print("=" * 60)
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    se_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=FL,
        metrics=["accuracy"],
    )
    hist_p3 = se_model.fit(
        gen_full,
        validation_data=val_gen,
        epochs=25,
        callbacks=get_callbacks_list(patience_es=8, patience_lr=4),
        verbose=1,
    )
    HISTORY["phase3"] = hist_p3.history
    print(f"✅ Phase 3 done | Best val acc: {max(hist_p3.history['val_accuracy']):.4f}")

    # ── 8. Save SE model ──────────────────────────────────────────────────────
    model_path = os.path.join(out_dir, "se_mobilenetv2_eurosat.h5")
    se_model.save(model_path)
    print(f"💾 Model saved: {model_path}")

    # ── 9. Evaluate SE model ──────────────────────────────────────────────────
    se_pred_proba = se_model.predict(X_test, verbose=0)
    se_pred = np.argmax(se_pred_proba, axis=1)
    se_acc = accuracy_score(y_test, se_pred) * 100
    se_f1 = f1_score(y_test, se_pred, average="macro")
    print(f"\n🏆 SE-MobileNetV2: Acc={se_acc:.2f}%  F1={se_f1:.4f}")

    # ── 10. Classical ML baselines ────────────────────────────────────────────
    print("\n⏳ Training classical ML baselines...")
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    ml_results = {}
    for name, clf in classifiers.items():
        clf.fit(X_sc_train, y_train)
        pred = clf.predict(X_sc_test)
        acc = accuracy_score(y_test, pred) * 100
        f1 = f1_score(y_test, pred, average="macro")
        ml_results[name] = {"acc": acc, "f1": f1}
        print(f"   ✅ {name}: Acc={acc:.1f}%  F1={f1:.3f}")

    # ── 11. Vanilla MobileNetV2 baseline ─────────────────────────────────────
    print("\n⏳ Training Vanilla MobileNetV2 baseline...")
    from tensorflow.keras.applications import MobileNetV2 as MNV2
    from tensorflow import keras as K2
    from tensorflow.keras import layers as L2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG

    base_v = MNV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_v.trainable = False
    inp_v = K2.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x_v = base_v(inp_v, training=False)
    x_v = L2.GlobalAveragePooling2D()(x_v)
    x_v = L2.Dense(256, activation="relu")(x_v)
    x_v = L2.Dropout(0.4)(x_v)
    x_v = L2.BatchNormalization()(x_v)
    out_v = L2.Dense(NUM_CLASSES, activation="softmax")(x_v)
    vanilla_mob = K2.Model(inp_v, out_v, name="Vanilla_MobileNetV2")

    aug_gen = IDG(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
        fill_mode="nearest",
    )
    vanilla_mob.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    vanilla_mob.fit(
        aug_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_test, y_test),
        epochs=15,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        callbacks=get_callbacks_list(patience_es=5, patience_lr=3),
        verbose=1,
    )
    base_v.trainable = True
    for layer in base_v.layers[:-30]:
        layer.trainable = False
    vanilla_mob.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    vanilla_mob.fit(
        aug_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_test, y_test),
        epochs=25,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        callbacks=get_callbacks_list(patience_es=6, patience_lr=3),
        verbose=1,
    )
    van_pred = np.argmax(vanilla_mob.predict(X_test, verbose=0), axis=1)
    van_acc = accuracy_score(y_test, van_pred) * 100
    van_f1 = f1_score(y_test, van_pred, average="macro")
    print(f"✅ Vanilla MobileNetV2: Acc={van_acc:.2f}%  F1={van_f1:.4f}")

    # ── 12. Agentic Router ────────────────────────────────────────────────────
    print("\n⏳ Building Agentic Router...")
    feature_extractor = build_feature_extractor(se_model)
    knn_fallback, emb_scaler = build_knn_fallback(feature_extractor, X_train, y_train)

    routed_preds, pred_sources, se_confidences, _ = agentic_router(
        X_test, se_model, knn_fallback, feature_extractor, emb_scaler
    )
    routed_acc = accuracy_score(y_test, routed_preds) * 100
    routed_f1 = f1_score(y_test, routed_preds, average="macro")
    n_se_direct = (pred_sources == "se_direct").sum()
    n_knn_routed = (pred_sources == "knn_fallback").sum()
    print(f"🤖 Agentic Router: Acc={routed_acc:.2f}%  F1={routed_f1:.4f}")
    print(f"   SE Direct: {n_se_direct}  KNN Routed: {n_knn_routed}")

    # ── 13. Clustering ────────────────────────────────────────────────────────
    print("\n⏳ Running clustering...")
    km = KMeans(n_clusters=NUM_CLASSES, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    km_labels = km.fit_predict(X_sc_train)
    ari = adjusted_rand_score(y_train, km_labels)
    nmi = normalized_mutual_info_score(y_train, km_labels)
    sil = silhouette_score(X_sc_train, km_labels, sample_size=800)

    kmed_labels = k_medoids_fit(X_sc_train, k=NUM_CLASSES)
    ari2 = adjusted_rand_score(y_train, kmed_labels)
    sil2 = silhouette_score(X_sc_train, kmed_labels, sample_size=800)
    print(f"K-Means: ARI={ari:.3f}  NMI={nmi:.3f}  Sil={sil:.3f}")
    print(f"K-Medoids: ARI={ari2:.3f}  Sil={sil2:.3f}")

    # ── 14. Save all artefacts ────────────────────────────────────────────────
    artefacts = {
        "classes": classes,
        "num_classes": NUM_CLASSES,
        "img_size": IMG_SIZE,
        "se_acc": se_acc,
        "se_f1": se_f1,
        "van_acc": van_acc,
        "van_f1": van_f1,
        "routed_acc": routed_acc,
        "routed_f1": routed_f1,
        "n_se_direct": int(n_se_direct),
        "n_knn_routed": int(n_knn_routed),
        "n_test": len(X_test),
        "avg_se_conf": float(se_confidences.mean()),
        "ml_results": ml_results,
        "clustering": {
            "km_ari": ari, "km_nmi": nmi, "km_sil": sil,
            "kmed_ari": ari2, "kmed_sil": sil2,
        },
        "history": {
            "phase1_val_acc": HISTORY["phase1"]["val_accuracy"],
            "phase2_val_acc": HISTORY["phase2"]["val_accuracy"],
            "phase3_val_acc": HISTORY["phase3"]["val_accuracy"],
        },
        "difficulty": difficulty.tolist(),
        "tier_easy_max": float(difficulty[tier_easy].max()),
        "tier_med_max": float(difficulty[tier_med].max()),
        "confidence_threshold": 0.60,
    }

    with open(os.path.join(out_dir, "artefacts.json"), "w") as f:
        json.dump(artefacts, f, indent=2)

    # Save sklearn objects
    with open(os.path.join(out_dir, "knn_fallback.pkl"), "wb") as f:
        pickle.dump(knn_fallback, f)
    with open(os.path.join(out_dir, "emb_scaler.pkl"), "wb") as f:
        pickle.dump(emb_scaler, f)
    with open(os.path.join(out_dir, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Save test data for dashboard visualisations
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)
    np.save(os.path.join(out_dir, "X_sc_train.npy"), X_sc_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)

    print(f"\n✅ All artefacts saved to: {out_dir}")
    print("\n" + "=" * 58)
    print(f"  {'Model':<32} {'Accuracy':>10}  {'Macro-F1':>10}")
    print("=" * 58)
    all_results = {
        **{k: v for k, v in ml_results.items()},
        "Vanilla MobileNetV2": {"acc": van_acc, "f1": van_f1},
        "SE-MobileNetV2 (Focal+Curr)": {"acc": se_acc, "f1": se_f1},
        "SE + Agentic Router": {"acc": routed_acc, "f1": routed_f1},
    }
    for rank, name in enumerate(
        sorted(all_results, key=lambda x: all_results[x]["acc"], reverse=True), 1
    ):
        star = "  ◀ BEST" if rank == 1 else ""
        print(
            f"  #{rank:<2} {name:<30} {all_results[name]['acc']:>9.2f}%  "
            f"{all_results[name]['f1']:>10.4f}{star}"
        )
    print("=" * 58)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SE-MobileNetV2 on EuroSAT")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to EuroSAT RGB dataset directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artefacts",
        help="Directory to save model and artefacts",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or find_dataset_dir()
    if data_dir is None:
        raise RuntimeError(
            "EuroSAT dataset not found. "
            "Pass --data_dir /path/to/EuroSAT_RGB or place data in ./data/EuroSAT"
        )

    print(f"📂 Dataset: {data_dir}")
    train(data_dir, args.out_dir)

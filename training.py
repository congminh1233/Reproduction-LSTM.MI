from typing import Dict, Any, Optional, List, Tuple
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)
from sklearn.utils import class_weight

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)
# -----------------------
# Utilities: tokenizer
# -----------------------
def build_char_vocab(domains: List[str], lower: bool = True) -> Dict[str, int]:
    """Build char->index mapping. reserve 0 for padding."""
    chars = set()
    for d in domains:
        if lower:
            d = d.lower()
        chars.update(list(d))
    chars = sorted(chars)
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    return stoi

def domain_to_seq(domain: str, stoi: Dict[str,int], maxlen: int, lower: bool = True) -> List[int]:
    if lower:
        domain = domain.lower()
    seq = [stoi.get(c, 0) for c in domain]
    return seq[:maxlen]

def domains_to_padded(domains: List[str], stoi: Dict[str,int], maxlen: int) -> np.ndarray:
    seqs = [domain_to_seq(d, stoi, maxlen) for d in domains]
    return pad_sequences(seqs, maxlen=maxlen, padding='post', truncating='post')

# -----------------------
# Cost helpers (paper)
# -----------------------
def compute_diag_costs_from_counts(counts: np.ndarray, gamma: float = 0.3) -> np.ndarray:
    """
    C[i] = (1 / n_i)^gamma
    counts: array-like of class sample counts (length = num_classes)
    """
    counts = np.array(counts, dtype=np.float32)
    counts[counts == 0] = 1.0
    costs = (1.0 / counts) ** gamma
    return costs

def build_full_cost_matrix_from_diagonal(diag_costs: np.ndarray) -> np.ndarray:
    """
    Build full cost matrix C where C[i, k] = diag_costs[i] if i==k else 1.0
    """
    n = len(diag_costs)
    C = np.ones((n, n), dtype=np.float32)
    for i in range(n):
        C[i, i] = float(diag_costs[i])
    return C

# -----------------------
# Loss functions
# -----------------------
def weighted_categorical_crossentropy_diag(class_costs: np.ndarray):
    """
    Loss = CE(y_true, y_pred) * cost_of_true_class
    class_costs: array shape (num_classes,)
    Implements Equation (7)-like behavior.
    """
    class_costs_tf = tf.constant(class_costs, dtype=tf.float32)
    def loss(y_true, y_pred):
        # y_true: one-hot
        base = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        per_sample_cost = tf.reduce_sum(y_true * class_costs_tf, axis=-1)
        return base * per_sample_cost
    return loss

def weighted_categorical_crossentropy_full(cost_matrix: np.ndarray):
    """
    Loss implementing Eq.(8) from paper:
      - sum_k t_k * log(p_k) * C[class(p), k]
    cost_matrix: shape (num_original_classes, num_predicted_classes)
    Requires y_true to be one-hot of original classes used to index rows of C.
    """
    C_tf = tf.constant(cost_matrix, dtype=tf.float32)
    eps = tf.keras.backend.epsilon()
    def loss(y_true, y_pred):
        # cost_row (batch, n_classes) = y_true @ C
        cost_row = tf.matmul(y_true, C_tf)
        logp = tf.math.log(tf.clip_by_value(y_pred, eps, 1.0))
        per_sample = - tf.reduce_sum(y_true * logp * cost_row, axis=-1)
        return per_sample
    return loss

# -----------------------
# Model factory
# -----------------------
def build_lstm_model(vocab_size: int, maxlen: int, num_classes: int,
                     embedding_dim: int = 128, lstm_units: int = 128, dropout: float = 0.2) -> Model:
    inp = Input(shape=(maxlen,), dtype='int32')
    emb = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=maxlen, mask_zero=True)(inp)
    x = LSTM(lstm_units)(emb)
    x = Dropout(dropout)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    return model

# -----------------------
# Metrics
# -----------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> Dict[str, float]:
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return {'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}

# -----------------------
# TRAINING pipeline
# -----------------------
def train_lstm_mi(df: pd.DataFrame,
                  label_map: Dict[str,int],
                  domain_col: str = 'domain',
                  label_col: str = 'label',
                  non_dga_label: int = 0,
                  maxlen: int = 64,
                  embedding_dim: int = 128,
                  lstm_units: int = 128,
                  gamma: float = 0.3,
                  batch_size: int = 16,
                  epochs: int = 15,
                  test_size: float = 0.2,
                  random_state: int = 42,
                  use_full_cost_matrix: bool = False,
                  save_dir: Optional[str] = None) -> Dict[str, Any]:

    # Prepare data
    domains = df[domain_col].astype(str).tolist()
    labels = df[label_col].astype(int).values

    # tokenizer
    stoi = build_char_vocab(domains)
    X_all = domains_to_padded(domains, stoi, maxlen)

    # ---------------- Binary stage ----------------
    y_bin = (labels != non_dga_label).astype(int)
    counts_bin = [int(np.sum(y_bin == 0)), int(np.sum(y_bin == 1))]
    diag_bin = compute_diag_costs_from_counts(np.array(counts_bin), gamma=gamma)
    loss_bin = weighted_categorical_crossentropy_diag(diag_bin)

    # train/val split (stratify by binary)
    X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(
        X_all, y_bin, test_size=test_size, random_state=random_state, stratify=y_bin
    )

    model_bin = build_lstm_model(vocab_size=len(stoi), maxlen=maxlen, num_classes=2,
                                 embedding_dim=embedding_dim, lstm_units=lstm_units)
    model_bin.compile(optimizer='adam', loss=loss_bin, metrics=['accuracy'])

    cb_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "checkpoint_best_model.h5",
        save_best_only=True,
        monitor="val_loss"
    )

    cb_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    print("[Train] Binary model (non-DGA vs DGA):")
    hist_bin = model_bin.fit(
        X_train_b,
        tf.keras.utils.to_categorical(y_train_b, num_classes=2),
        validation_data=(X_val_b, tf.keras.utils.to_categorical(y_val_b, num_classes=2)),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cb_early, cb_checkpoint, cb_lr],
        verbose=2
    )

    # ---------------- Multiclass stage (DGA-only) ----------------
    mask_dga = (labels != non_dga_label)
    X_dga = X_all[mask_dga]
    y_dga = labels[mask_dga]

    if len(y_dga) == 0:
        raise ValueError("No DGA samples found in dataset for multiclass training.")

    # Reindex original labels (which might be sparse) to dense 0..K-1 for softmax
    unique_orig = np.unique(y_dga)
    label_to_dense = {orig: i for i, orig in enumerate(unique_orig)}
    dense_to_label = {i: int(orig) for orig, i in label_to_dense.items()}
    y_dga_dense = np.array([label_to_dense[int(v)] for v in y_dga], dtype=int)

    # compute diag costs for multiclass
    counts_dga = [int(np.sum(y_dga_dense == i)) for i in range(len(unique_orig))]
    diag_multi = compute_diag_costs_from_counts(np.array(counts_dga), gamma=gamma)

    if use_full_cost_matrix:
        C_full = build_full_cost_matrix_from_diagonal(diag_multi)
        loss_multi = weighted_categorical_crossentropy_full(C_full)
    else:
        loss_multi = weighted_categorical_crossentropy_diag(diag_multi)

    # split DGA multiclass
    X_train_m, X_val_m, y_train_m, y_val_m = train_test_split(
        X_dga, y_dga_dense, test_size=test_size, random_state=random_state, stratify=y_dga_dense
    )

    model_multi = build_lstm_model(vocab_size=len(stoi), maxlen=maxlen, num_classes=len(unique_orig),
                                   embedding_dim=embedding_dim, lstm_units=lstm_units)
    model_multi.compile(optimizer='adam', loss=loss_multi, metrics=['accuracy'])

    print(f"[Train] Multiclass model on {len(unique_orig)} DGA classes:")
    hist_multi = model_multi.fit(
        X_train_m,
        tf.keras.utils.to_categorical(y_train_m, num_classes=len(unique_orig)),
        validation_data=(X_val_m, tf.keras.utils.to_categorical(y_val_m, num_classes=len(unique_orig))),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cb_early, cb_checkpoint, cb_lr],
        verbose=2
    )

    # ---------------- Save artifacts if requested ----------------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_bin.save(os.path.join(save_dir, "lstmmi_binary.h5"))
        model_multi.save(os.path.join(save_dir, "lstmmi_multiclass.h5"))
        with open(os.path.join(save_dir, "stoi.json"), "w", encoding='utf-8') as f:
            json.dump(stoi, f, ensure_ascii=False, indent=2)
        # Save label_map (family->original_label)
        with open(os.path.join(save_dir, "label_map_TLD.json"), "w", encoding='utf-8') as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
        # Save dense_to_label mapping (dense idx -> original label)
        with open(os.path.join(save_dir, "dense_map.json"), "w", encoding='utf-8') as f:
            json.dump(dense_to_label, f, indent=2)
        print(f"[Save] models and maps saved to: {save_dir}")

    # ---------------- Return pipeline ----------------
    pipeline = {
        "stoi": stoi,
        "model_bin": model_bin,
        "model_multi": model_multi,
        "label_map": label_map,
        "label_to_dense": label_to_dense,
        "dense_to_label": dense_to_label,
        "hist_bin": hist_bin.history,
        "hist_multi": hist_multi.history,
        "diag_bin": diag_bin,
        "diag_multi": diag_multi
    }
    return pipeline

# -----------------------
# TRAINING K-fold
# -----------------------
def train_lstm_mi_kfold(
    df,
    domain_col="domain",
    label_col="label",
    non_dga_label=0,
    n_splits=10,
    maxlen=64,
    embedding_dim=128,
    lstm_units=128,
    gamma=0.3,
    batch_size=512,
    epochs=20,
    random_state=42,
):

    domains = df[domain_col].astype(str).tolist()
    labels = df[label_col].astype(int).values

    stoi = build_char_vocab(domains)
    X = domains_to_padded(domains, stoi, maxlen)
    y_bin = (labels != non_dga_label).astype(int)

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    fold_f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_bin), 1):
        print(f"\n===== Fold {fold}/{n_splits} =====")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        y_bin_train = y_bin[train_idx]

        # ===============================
        # Binary stage
        # ===============================
        counts_bin = [
            np.sum(y_bin_train == 0),
            np.sum(y_bin_train == 1),
        ]
        loss_bin = weighted_categorical_crossentropy_diag(
            compute_diag_costs_from_counts(counts_bin, gamma)
        )

        model_bin = build_lstm_model(
            vocab_size=len(stoi),
            maxlen=maxlen,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            num_classes=2,
        )
        model_bin.compile(optimizer="adam", loss=loss_bin)

        model_bin.fit(
            X_train,
            tf.keras.utils.to_categorical(y_bin_train, 2),
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=3, restore_best_weights=True
                )
            ],
            verbose=0,
        )

        bin_pred = np.argmax(
            model_bin.predict(X_test, batch_size=1024, verbose=0),
            axis=1,
        )

        # ===============================
        # Multiclass stage (DGA only)
        # ===============================
        dga_mask_train = y_train != non_dga_label
        X_dga_train = X_train[dga_mask_train]
        y_dga_train = y_train[dga_mask_train]

        uniq = np.unique(y_dga_train)
        label_to_dense = {v: i for i, v in enumerate(uniq)}
        dense_to_label = {i: v for v, i in label_to_dense.items()}
        y_dense = np.array([label_to_dense[v] for v in y_dga_train])

        loss_multi = weighted_categorical_crossentropy_diag(
            compute_diag_costs_from_counts(
                [np.sum(y_dense == i) for i in range(len(uniq))],
                gamma,
            )
        )

        model_multi = build_lstm_model(
            vocab_size=len(stoi),
            maxlen=maxlen,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            num_classes=len(uniq),
        )
        model_multi.compile(optimizer="adam", loss=loss_multi)

        model_multi.fit(
            X_dga_train,
            tf.keras.utils.to_categorical(y_dense, len(uniq)),
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=3, restore_best_weights=True
                )
            ],
            verbose=0,
        )

        # ===============================
        # Final prediction
        # ===============================
        final_preds = np.zeros(len(X_test), dtype=int)
        final_preds[bin_pred == 0] = non_dga_label

        dga_mask_test = bin_pred == 1
        if np.any(dga_mask_test):
            preds = model_multi.predict(
                X_test[dga_mask_test], batch_size=1024, verbose=0
            )
            dense_idx = np.argmax(preds, axis=1)
            final_preds[dga_mask_test] = [
                dense_to_label[i] for i in dense_idx
            ]

        # ===============================
        # Evaluation
        # ===============================
        print("\nClassification report:")
        report_txt = classification_report(
            y_test, final_preds, digits=4, zero_division=0
        )
        print(report_txt)

        precision = precision_score(
            y_test, final_preds, average="macro", zero_division=0
        )
        recall = recall_score(
            y_test, final_preds, average="macro", zero_division=0
        )
        f1 = f1_score(
            y_test, final_preds, average="macro", zero_division=0
        )
        acc = accuracy_score(y_test, final_preds)

        print(f"F1 score : {f1:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Acc      : {acc:.4f}")

        classifaction_report_csv(
            report_txt, precision, recall, f1, fold
        )

        fold_f1_scores.append(f1)

    print("\n===== FINAL K-FOLD RESULT =====")
    print(f"Macro-F1 (mean): {np.mean(fold_f1_scores):.4f}")
    print(f"Macro-F1 (std) : {np.std(fold_f1_scores):.4f}")

def classifaction_report_csv(
    report: str,
    precision: float,
    recall: float,
    f1_score: float,
    fold: int,
    filename: str = "classification_report_cost.csv",
):

    import pandas as pd
    import os

    report_data = []

    report_data.append({
        "class": f"fold {fold}",
        "precision": "",
        "recall": "",
        "f1_score": "",
        "support": "",
    })

    lines = report.split("\n")

    for line in lines[2:]:
        line = " ".join(line.strip().split())
        if not line:
            continue

        row = line.split(" ")
        if len(row) < 4:
            continue

        # avg / total
        if row[0] == "avg" or row[0] == "macro" or row[0] == "weighted":
            continue

        try:
            report_data.append({
                "class": row[0],
                "precision": float(row[1]),
                "recall": float(row[2]),
                "f1_score": float(row[3]),
                "support": row[4] if len(row) > 4 else "",
            })
        except ValueError:
            continue

    report_data.append({
        "class": "macro",
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "support": 0,
    })

    df = pd.DataFrame(report_data)

    write_header = not os.path.exists(filename)
    df.to_csv(filename, mode="a", header=write_header, index=False)

# -----------------------
# Inference pipeline
# -----------------------
def lstm_mi_predict(domains: List[str], pipeline: Dict[str,Any], maxlen: int = 64) -> List[Dict[str,Any]]:
    """
    pipeline: returned dict from train_lstm_mi OR load_lstm_mi
    Returns list of dicts with keys:
      domain, type ('non-DGA' or 'DGA'), label (original integer), family (string), prob (for DGA)
    """
    stoi = pipeline["stoi"]
    model_bin = pipeline["model_bin"]
    model_multi = pipeline["model_multi"]
    label_map = pipeline["label_map"]
    dense_to_label = pipeline["dense_to_label"]

    # reverse label_map: original label -> family
    label_to_family = {int(v): k for k, v in label_map.items()}

    X = domains_to_padded(domains, stoi, maxlen)
    pred_bin = model_bin.predict(X)
    bin_labels = np.argmax(pred_bin, axis=1)

    outputs = []
    for i, d in enumerate(domains):
        if bin_labels[i] == 0:
            outputs.append({
                "domain": d,
                "type": "non-DGA",
                "label": 0,
                "family": "non-dga"
            })
        else:
            proba = model_multi.predict(X[i:i+1], verbose=0)[0]
            dense_idx = int(np.argmax(proba))
            orig_label = int(dense_to_label[dense_idx])
            fam = label_to_family.get(orig_label, "unknown")
            outputs.append({
                "domain": d,
                "type": "DGA",
                "label": orig_label,
                "family": fam,
                "prob": proba.tolist()
            })
    return outputs

# -----------------------
# Save/load pipeline utilities
# -----------------------
def load_lstm_mi(save_dir: str) -> Dict[str,Any]:
    """
    Load saved models + maps from save_dir (expects files saved by train_lstm_mi)
    """
    with open(os.path.join(save_dir, "stoi.json"), "r", encoding='utf-8') as f:
        stoi = json.load(f)
    with open(os.path.join(save_dir, "label_map_TLD.json"), "r", encoding='utf-8') as f:
        label_map = json.load(f)
    with open(os.path.join(save_dir, "dense_map.json"), "r", encoding='utf-8') as f:
        dense_to_label = {int(k): int(v) for k, v in json.load(f).items()}

    model_bin = load_model(os.path.join(save_dir, "lstmmi_binary.h5"), compile=False)
    model_multi = load_model(os.path.join(save_dir, "lstmmi_multiclass.h5"), compile=False)

    label_to_dense = {int(v): int(k) for k, v in dense_to_label.items()}

    pipeline = {
        "stoi": stoi,
        "model_bin": model_bin,
        "model_multi": model_multi,
        "label_map": label_map,
        "label_to_dense": label_to_dense,
        "dense_to_label": dense_to_label
    }
    return pipeline

if __name__ == "__main__":
    # Config
    DATASET_CSV = "dataset_TLD.csv"
    LABEL_MAP_JSON = "label_map_TLD.json"
    SAVE_DIR = "saved_model"

    # Load dataset + label_map
    df = pd.read_csv(DATASET_CSV)
    with open(LABEL_MAP_JSON, "r", encoding='utf-8') as f:
        label_map = json.load(f)

    # train_lstm_mi_kfold(
    #     df,
    #     n_splits=10,
    #     maxlen=64,
    #     embedding_dim=128,
    #     lstm_units=128,
    #     gamma=0.3,
    #     batch_size=256,
    #     epochs=40,
    #     random_state=42
    # )

    # Train & export model
    pipeline = train_lstm_mi(
        df=df,
        label_map=label_map,
        maxlen=64,
        gamma=0.3,
        embedding_dim=128,
        lstm_units=128,
        epochs=40,
        batch_size=256,
        test_size=0.2,
        use_full_cost_matrix=False,
        save_dir=SAVE_DIR
    )
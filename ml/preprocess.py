"""Preprocessing + partition UCI Diabetes into 3 hospitals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "diabetic_data.csv"
HOSP_DIRS = {
    1: ROOT / "data" / "hospital_1",
    2: ROOT / "data" / "hospital_2",
    3: ROOT / "data" / "hospital_3",
}
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "ml" / "artifacts"

NUMERIC_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
CATEGORICAL_FEATURES = [
    "race", "gender", "age", "admission_type_id",
    "discharge_disposition_id", "admission_source_id",
    "max_glu_serum", "A1Cresult", "metformin", "insulin",
    "change", "diabetesMed",
]
TARGET = "readmitted"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, low_memory=False)
    print(f"[preprocess] Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.replace("?", np.nan)
    df[TARGET] = (df[TARGET] == "<30").astype(int)
    df = df.dropna(subset=["race", "gender", "age"])
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
    print(f"[preprocess] After clean: {len(df)} rows, positive rate={df[TARGET].mean():.3f}")
    return df


def encode_and_scale(df: pd.DataFrame):
    encoders = {}
    X = pd.DataFrame(index=df.index)
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        X[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    for col in NUMERIC_FEATURES:
        X[col] = df[col].astype(float)
    scaler = StandardScaler()
    X[NUMERIC_FEATURES] = scaler.fit_transform(X[NUMERIC_FEATURES])
    y = df[TARGET].values
    return X, y, encoders, scaler


def hospital_partition(df: pd.DataFrame) -> dict:
    y = df[TARGET].values
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    partitions = {}
    for fold_idx, (_, test_idx) in enumerate(skf.split(df, y)):
        h = fold_idx + 1
        part = df.iloc[test_idx].copy()
        partitions[h] = part
        print(f"[partition] Hospital {h}: {len(part)} samples (pos_rate={part[TARGET].mean():.3f})")
    return partitions


def save_hospital_data(partitions: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    full = pd.concat(partitions.values(), ignore_index=True)
    X_full, y_full, encoders, scaler = encode_and_scale(full)

    joblib.dump(encoders, ARTIFACTS / "encoders.joblib")
    joblib.dump(scaler, ARTIFACTS / "scaler.joblib")
    feature_names = list(X_full.columns)
    (ARTIFACTS / "feature_names.json").write_text(json.dumps(feature_names))

    meta = {
        "hospitals": {},
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "partition_method": "StratifiedKFold(n_splits=3, shuffle=True, random_state=42)",
        "target": "readmitted_<30_days_binary",
    }

    for h, part in partitions.items():
        out_dir = HOSP_DIRS[h]
        out_dir.mkdir(parents=True, exist_ok=True)

        X = pd.DataFrame(index=part.index)
        for col in CATEGORICAL_FEATURES:
            le = encoders[col]
            vals = part[col].astype(str)
            known = set(le.classes_)
            vals = vals.apply(lambda v: v if v in known else le.classes_[0])
            X[col] = le.transform(vals)
        for col in NUMERIC_FEATURES:
            X[col] = part[col].astype(float)
        X[NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])
        y = part[TARGET].values

        X_train, X_test, y_train, y_test = train_test_split(
            X.values.astype(np.float32),
            y.astype(np.float32),
            test_size=0.2,
            random_state=42 + h,
            stratify=y,
        )

        np.savez_compressed(out_dir / "train.npz", X=X_train, y=y_train)
        np.savez_compressed(out_dir / "test.npz", X=X_test, y=y_test)
        part.head(200).to_csv(out_dir / "sample.csv", index=False)

        meta["hospitals"][h] = {
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "pos_rate_train": float(y_train.mean()),
            "pos_rate_test": float(y_test.mean()),
        }
        print(f"[save] Hospital {h}: train={len(y_train)}, test={len(y_test)}, pos={y_train.mean():.3f}")

    (PROCESSED / "partition_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[preprocess] Meta -> {PROCESSED / 'partition_meta.json'}")


def main():
    df = load_raw()
    df = clean(df)
    partitions = hospital_partition(df)
    save_hospital_data(partitions)
    print("[preprocess] Done.")


if __name__ == "__main__":
    main()
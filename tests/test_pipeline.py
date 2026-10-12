"""Tests du pipeline Smart Healthcare Data Space."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_dataset_exists():
    raw = ROOT / "data" / "raw" / "diabetic_data.csv"
    assert raw.exists(), "diabetic_data.csv manquant"
    assert raw.stat().st_size > 10_000_000, "dataset trop petit"


def test_hospital_partitions():
    for h in (1, 2, 3):
        train = ROOT / "data" / f"hospital_{h}" / "train.npz"
        test = ROOT / "data" / f"hospital_{h}" / "test.npz"
        assert train.exists(), f"hospital {h} train manquant"
        assert test.exists(), f"hospital {h} test manquant"
        d = np.load(train)
        assert "X" in d and "y" in d
        assert d["X"].shape[1] == 20
        assert len(d["y"]) > 1000


def test_partitions_not_identical():
    sizes = []
    for h in (1, 2, 3):
        d = np.load(ROOT / "data" / f"hospital_{h}" / "train.npz")
        sizes.append(len(d["y"]))
    # tailles proches mais jeux distincts (fichiers différents)
    assert all(s > 1000 for s in sizes)


def test_model_forward():
    from ml.model import get_model, get_parameters, set_parameters
    m = get_model(20)
    x = torch.randn(4, 20)
    y = m(x)
    assert y.shape == (4,)
    params = get_parameters(m)
    assert len(params) > 0
    set_parameters(m, params)


def test_training_results_real():
    hist_path = ROOT / "results" / "training_history.json"
    assert hist_path.exists(), "Lancer run_fedavg.py d'abord"
    hist = json.loads(hist_path.read_text(encoding="utf-8"))
    assert hist["num_rounds"] >= 1
    assert len(hist["history"]) >= 1
    last = hist["history"][-1]
    assert 0.5 < last["aggregated_accuracy"] < 1.0
    assert last["aggregated_loss"] > 0
    for h in ("1", "2", "3"):
        assert h in last["evaluate"]


def test_partition_meta():
    meta = json.loads((ROOT / "data" / "processed" / "partition_meta.json").read_text(encoding="utf-8"))
    assert meta["n_features"] == 20
    assert "hospitals" in meta


if __name__ == "__main__":
    tests = [
        test_dataset_exists,
        test_hospital_partitions,
        test_partitions_not_identical,
        test_model_forward,
        test_training_results_real,
        test_partition_meta,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
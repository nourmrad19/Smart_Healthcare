"""FastAPI backend for Smart Healthcare Data Space."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
DASHBOARD = ROOT / "dashboard"

app = FastAPI(
    title="Smart Healthcare Data Space",
    description="Federated Learning + EDC platform for 3 hospitals",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EDC_STATE = {
    "connectors": {
        "edc-hospital-1": {
            "id": "edc-hospital-1",
            "participant": "Hospital 1",
            "status": "online",
            "catalog_items": 2,
            "active_contracts": 1,
            "last_negotiation": "model-params-exchange",
            "endpoint": "http://edc1:19193/api",
        },
        "edc-hospital-2": {
            "id": "edc-hospital-2",
            "participant": "Hospital 2",
            "status": "online",
            "catalog_items": 2,
            "active_contracts": 1,
            "last_negotiation": "model-params-exchange",
            "endpoint": "http://edc2:19194/api",
        },
        "edc-hospital-3": {
            "id": "edc-hospital-3",
            "participant": "Hospital 3",
            "status": "online",
            "catalog_items": 2,
            "active_contracts": 1,
            "last_negotiation": "model-params-exchange",
            "endpoint": "http://edc3:19195/api",
        },
    },
    "negotiations": [
        {
            "id": "neg-001",
            "provider": "edc-hospital-1",
            "consumer": "edc-hospital-2",
            "asset": "model-weights-round-N",
            "state": "FINALIZED",
            "note": "Only model parameters exchanged – no raw patient data",
        },
        {
            "id": "neg-002",
            "provider": "edc-hospital-2",
            "consumer": "edc-hospital-3",
            "asset": "model-weights-round-N",
            "state": "FINALIZED",
            "note": "Only model parameters exchanged – no raw patient data",
        },
    ],
    "catalog": [
        {
            "asset_id": "fl-model-params",
            "name": "Federated Model Parameters",
            "description": "PyTorch state dict (weights only). NO patient data.",
            "policy": "usage-policy-fl-only",
            "owner": "Flower Server via EDC",
        },
        {
            "asset_id": "training-metrics",
            "name": "Aggregated Training Metrics",
            "description": "Accuracy / loss per round (no raw data)",
            "policy": "read-only-metrics",
            "owner": "Flower Server",
        },
    ],
}


def load_training_history() -> Dict[str, Any]:
    path = RESULTS / "training_history.json"
    if not path.exists():
        return {"num_rounds": 0, "history": [], "status": "not_started"}
    return json.loads(path.read_text(encoding="utf-8"))


def load_partition_meta() -> Dict[str, Any]:
    path = DATA / "processed" / "partition_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "smart-healthcare-dataspace",
        "timestamp": time.time(),
    }


@app.get("/hospitals")
def hospitals():
    meta = load_partition_meta()
    hist = load_training_history()
    last_eval = {}
    if hist.get("history"):
        last_eval = hist["history"][-1].get("evaluate", {})

    result = []
    for h in (1, 2, 3):
        hmeta = meta.get("hospitals", {}).get(str(h), meta.get("hospitals", {}).get(h, {}))
        eval_h = last_eval.get(str(h), {})
        result.append(
            {
                "id": h,
                "name": f"Hospital {h}",
                "status": "online",
                "n_train": hmeta.get("n_train"),
                "n_test": hmeta.get("n_test"),
                "pos_rate_train": hmeta.get("pos_rate_train"),
                "last_accuracy": eval_h.get("accuracy"),
                "last_loss": eval_h.get("loss"),
                "data_location": f"data/hospital_{h}/ (local only)",
                "edc_connector": f"edc-hospital-{h}",
            }
        )
    return {"hospitals": result}


@app.get("/training/status")
def training_status():
    hist = load_training_history()
    if not hist.get("history"):
        return {"status": "idle", "rounds_completed": 0}
    return {
        "status": "completed",
        "rounds_completed": hist.get("num_rounds", len(hist["history"])),
        "algorithm": hist.get("algorithm", "FedAvg"),
        "model": hist.get("model", "DiabetesNet"),
        "total_seconds": hist.get("total_seconds"),
        "final_accuracy": hist.get("final_aggregated_accuracy"),
        "final_loss": hist.get("final_aggregated_loss"),
    }


@app.get("/training/metrics")
def training_metrics():
    hist = load_training_history()
    if not hist.get("history"):
        raise HTTPException(404, "No training results yet. Run federated_learning/run_fedavg.py")
    rounds = []
    for entry in hist["history"]:
        rounds.append(
            {
                "round": entry["round"],
                "aggregated_accuracy": entry.get("aggregated_accuracy"),
                "aggregated_loss": entry.get("aggregated_loss"),
                "elapsed_seconds": entry.get("elapsed_seconds"),
                "per_hospital": entry.get("evaluate", {}),
                "fit": entry.get("fit", {}),
            }
        )
    return {
        "num_rounds": hist.get("num_rounds"),
        "total_seconds": hist.get("total_seconds"),
        "rounds": rounds,
        "backend": hist.get("backend"),
    }


@app.get("/edc/status")
def edc_status():
    return {
        "note": "EDC manages governance of model-parameter exchange only. "
                "Raw patient data stays local at each hospital.",
        "connectors": list(EDC_STATE["connectors"].values()),
        "active_negotiations": EDC_STATE["negotiations"],
        "catalog": EDC_STATE["catalog"],
    }


@app.get("/edc/catalog")
def edc_catalog():
    return {"catalog": EDC_STATE["catalog"]}


@app.get("/edc/negotiations")
def edc_negotiations():
    return {"negotiations": EDC_STATE["negotiations"]}


@app.get("/system/status")
def system_status():
    hist = load_training_history()
    meta = load_partition_meta()
    return {
        "platform": "Smart Healthcare Data Space",
        "components": {
            "flower_server": "ready",
            "flower_clients": 3,
            "edc_connectors": 3,
            "fastapi": "online",
            "dashboard": "online",
        },
        "dataset": {
            "source": "UCI Diabetes 130-US Hospitals 1999-2008",
            "n_features": meta.get("n_features"),
            "hospitals": meta.get("hospitals"),
        },
        "training": {
            "status": "completed" if hist.get("history") else "not_started",
            "rounds": hist.get("num_rounds", 0),
            "final_accuracy": hist.get("final_aggregated_accuracy"),
        },
        "privacy": {
            "raw_data_shared": False,
            "only_model_parameters_exchanged": True,
            "edc_role": "governance of parameter exchange contracts",
        },
    }


@app.get("/", response_class=HTMLResponse)
def root():
    index = DASHBOARD / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Smart Healthcare Data Space API</h1><p>See /docs</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
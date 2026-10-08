"""
mlops_utils.py — small helpers shared by the MLOps Monitoring notebooks.

You do NOT need to edit this file. It keeps the notebooks short by handling:
  * loading the HDB resale data (from the local copy, or downloading it)
  * adding the cleaned feature columns
  * building the same model pipeline every time
  * a tiny file-based "model registry" (used in notebook 03)
"""
import json
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "hdb_resale.csv")
MODELS_DIR = os.path.join(HERE, "models")

# data.gov.sg dataset: "Resale flat prices based on registration date from Jan-2017 onwards"
DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"

# The features our "production" model uses
NUM = ["floor_area_sqm", "remaining_lease_years", "storey"]
CAT = ["town", "flat_type", "flat_model"]
FEATURES = NUM + CAT
TARGET = "resale_price"

# The day our model "went live"
DEPLOY_DATE = "2022-01-01"


def download_hdb(path=DATA_PATH):
    """Download the latest resale data from data.gov.sg and save a local copy."""
    import requests

    base = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}"
    requests.get(base + "/initiate-download", timeout=60)
    url = None
    for _ in range(20):
        reply = requests.get(base + "/poll-download", timeout=60).json()
        url = reply.get("data", {}).get("url")
        if url:
            break
        time.sleep(3)
    if not url:
        raise RuntimeError("data.gov.sg did not return a download link — try again in a minute.")
    df = pd.read_csv(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_hdb(refresh=False):
    """Load HDB resale transactions with clean feature columns added.

    refresh=True downloads the newest data from data.gov.sg first.
    """
    if refresh or not os.path.exists(DATA_PATH):
        raw = download_hdb()
    else:
        raw = pd.read_csv(DATA_PATH)
    return add_features(raw)


def add_features(raw):
    df = raw.copy()
    df["date"] = pd.to_datetime(df["month"])
    # "10 TO 12" -> 11 (middle storey of the range)
    df["storey"] = df["storey_range"].str[:2].astype(int) + 1
    # "61 years 04 months" -> 61.33
    years = df["remaining_lease"].str.extract(r"(\d+)\s*year")[0].astype(float)
    months = df["remaining_lease"].str.extract(r"(\d+)\s*month")[0].astype(float).fillna(0)
    df["remaining_lease_years"] = years + months / 12
    # Drop the current (incomplete) month so monthly numbers are fair
    latest = df["month"].max()
    if df.loc[df["month"] == latest].shape[0] < 1000:
        df = df[df["month"] != latest]
    return df.reset_index(drop=True)


def make_model():
    """The model pipeline used in all notebooks (one-hot categories + gradient boosting)."""
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)],
        remainder="passthrough",
    )
    return Pipeline([
        ("prep", prep),
        ("model", HistGradientBoostingRegressor(max_iter=300, random_state=42)),
    ])


def score(y_true, y_pred):
    """The three numbers we watch: MAE (S$), MAPE (%), bias (S$, + means over-predict)."""
    err = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "MAPE_%": float(np.mean(np.abs(err) / np.asarray(y_true)) * 100),
        "bias": float(np.mean(err)),
    }


# ---------------------------------------------------------------------------
# A tiny file-based model registry (a "mini MLflow") used in notebook 03
# ---------------------------------------------------------------------------
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")


def _read_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"champion": None, "versions": {}}


def _write_registry(reg):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


def register_model(model, version, notes="", metrics=None, trained_until=None):
    """Save a model file and record it in the registry (does NOT make it champion)."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{version}.joblib")
    joblib.dump(model, path)
    reg = _read_registry()
    reg["versions"][version] = {
        "file": f"{version}.joblib",
        "registered_at": datetime.now().isoformat(timespec="seconds"),
        "trained_until": trained_until,
        "metrics": metrics or {},
        "notes": notes,
    }
    _write_registry(reg)
    return path


def set_champion(version):
    """Point production at a version. Rolling back = calling this with an older version."""
    reg = _read_registry()
    if version not in reg["versions"]:
        raise ValueError(f"{version} is not registered")
    reg.setdefault("history", []).append(
        {"at": datetime.now().isoformat(timespec="seconds"), "from": reg["champion"], "to": version}
    )
    reg["champion"] = version
    _write_registry(reg)


def load_champion():
    reg = _read_registry()
    if reg["champion"] is None:
        raise ValueError("No champion yet — register a model and call set_champion() first.")
    return reg["champion"], joblib.load(os.path.join(MODELS_DIR, reg["versions"][reg["champion"]]["file"]))


def show_registry():
    reg = _read_registry()
    rows = []
    for v, info in reg["versions"].items():
        rows.append({
            "version": v,
            "champion": "⭐" if v == reg["champion"] else "",
            "trained_until": info.get("trained_until"),
            **{k: round(val, 2) for k, val in info.get("metrics", {}).items()},
            "notes": info.get("notes", ""),
        })
    return pd.DataFrame(rows)


def reset_registry():
    """Start the registry from scratch (deletes saved models)."""
    if os.path.isdir(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            os.remove(os.path.join(MODELS_DIR, f))


def load_version(version):
    """Load one specific registered version (e.g. "v1"), whether or not it is live."""
    reg = _read_registry()
    if version not in reg["versions"]:
        raise ValueError(f"{version} is not registered")
    return joblib.load(os.path.join(MODELS_DIR, reg["versions"][version]["file"]))

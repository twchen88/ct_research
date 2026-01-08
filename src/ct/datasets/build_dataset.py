from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd
import yaml

from ct.datasets.history import BuildHistory
from ct.datasets.filtering import FilterHistory
from ct.datasets.featurization import FeaturizeHistory

from ct.utils.metadata import get_git_commit_hash
from ct.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DatasetArtifact:
    dataset_id: str
    artifact_dir: str
    dataset_path: str
    schema_path: str
    features_path: str
    meta_path: str
    done_path: str


def _stable_json_dumps(obj: Any) -> str:
    """Stable serialization for hashing: order keys, avoid nondeterministic repr."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _compute_dataset_id(cfg: Dict[str, Any], raw_snapshot_id: str) -> Tuple[str, str]:
    """
    Returns (dataset_id, cfg_hash).
    dataset_id should change iff cfg or raw_snapshot_id changes.
    """
    cfg_hash = _hash_str(_stable_json_dumps(cfg))
    dataset_id = f"ds_{_hash_str(_stable_json_dumps({'cfg_hash': cfg_hash, 'raw_snapshot_id': raw_snapshot_id}))[:10]}"
    return dataset_id, cfg_hash


def _read_raw_snapshot_csv(raw_snapshot_dir: Path, filename: str, *, max_rows: Optional[int] = None) -> pd.DataFrame:
    path = raw_snapshot_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Raw snapshot file not found: {path}")
    return pd.read_csv(path, nrows=max_rows)


def _infer_features_and_masks(
    df: pd.DataFrame,
    *,
    feature_prefixes: Sequence[str],
    mask_prefixes: Sequence[str],
) -> Tuple[list[str], list[str]]:
    features = [c for c in df.columns if any(c.startswith(p) for p in feature_prefixes)]
    masks = [c for c in df.columns if any(c.startswith(p) for p in mask_prefixes)]
    features.sort()
    masks.sort()
    return features, masks


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _write_done(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n")


def build_dataset(
    *,
    cfg: Dict[str, Any],
    raw_snapshot_dir: Path,
    raw_snapshot_id: str,
    artifact_root: Path,
    config_path: Optional[str] = None,
    force: bool = False,
) -> DatasetArtifact:
    """
    Build a dataset artifact from a raw snapshot directory.

    Inputs:
      - cfg: dataset stage config (already validated & resolved)
      - raw_snapshot_dir: concrete directory, e.g. data/raw/snapshots/<snapshot_id>
      - raw_snapshot_id: concrete id (NOT "latest")
      - artifact_root: base artifact root, e.g. artifacts/
      - config_path: optional path to the stage config file for provenance
      - force: if True, rebuild even if artifact exists

    Outputs:
      - DatasetArtifact: paths + dataset_id
    """
    # ----- 0) Compute identities and paths -----
    dataset_id, cfg_hash = _compute_dataset_id(cfg, raw_snapshot_id)
    artifact_dir = artifact_root / "datasets" / dataset_id

    dataset_path = artifact_dir / cfg.get("output", {}).get("filename", "dataset.parquet")
    meta_path = artifact_dir / "meta.yaml"
    schema_path = artifact_dir / "schema.yaml"
    features_path = artifact_dir / "features.json"
    done_path = artifact_dir / "_DONE"

    # If already built, reuse unless forced
    if not force and done_path.exists() and dataset_path.exists() and meta_path.exists():
        return DatasetArtifact(
            dataset_id=dataset_id,
            artifact_dir=str(artifact_dir),
            dataset_path=str(dataset_path),
            schema_path=str(schema_path),
            features_path=str(features_path),
            meta_path=str(meta_path),
            done_path=str(done_path),
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1) Load raw snapshot data -----
    input_filename = cfg["input"]["filename"]
    raw_df = _read_raw_snapshot_csv(raw_snapshot_dir, input_filename, max_rows=cfg["input"].get("max_rows"))

    # ----- 2) Run stage transforms -----
    history_cfg = cfg.get("history", {}) or {}
    time_bin_col = history_cfg.get("time_bin_col", "step_index")  # source of truth

    # Inject time_bin_col into later stages WITHOUT requiring YAML duplication
    filtering_cfg = {**(cfg.get("filtering", {}) or {}), "time_bin_col": time_bin_col}
    featurization_cfg = {**(cfg.get("featurization", {}) or {}), "time_bin_col": time_bin_col}

    history_df = BuildHistory(raw_df, history_cfg=history_cfg, input_schema_cfg=cfg.get("input_schema")).run()
    filtered_df = FilterHistory(history_df, filtering_cfg=filtering_cfg).run()

    featurizer = FeaturizeHistory(filtered_df, featurization_cfg=featurization_cfg)
    X = featurizer.run()  # expected MultiIndex (patient_id, time_bin_col)

    # ----- 2.5) Output index representation -----
    output_schema = cfg.get("output_schema", {}) or {}
    id_col = output_schema.get("id_col", "patient_id")
    time_col = output_schema.get("time_col", time_bin_col)  # usually step_index
    storage_mode = (output_schema.get("storage", {}) or {}).get("index_representation", "columns")

    if storage_mode == "columns":
        # reset index -> columns
        X_out = X.reset_index()

        # Make sure id/time column names match the contract
        # If your contract wants different names, rename here.
        rename_map: Dict[str, str] = {}
        if "patient_id" in X_out.columns and id_col != "patient_id":
            rename_map["patient_id"] = id_col
        if time_bin_col in X_out.columns and time_col != time_bin_col:
            rename_map[time_bin_col] = time_col

        if rename_map:
            X_out = X_out.rename(columns=rename_map)

        # Validate
        if id_col not in X_out.columns or time_col not in X_out.columns:
            raise ValueError(
                f"After reset_index()/rename, expected id_col={id_col!r} and time_col={time_col!r} "
                f"but got columns: {list(X_out.columns)[:40]} ..."
            )

    elif storage_mode == "multiindex":
        # Validate index names and (optionally) rename to contract
        X_out = X
        idx_names = list(X_out.index.names[:2])
        if idx_names != ["patient_id", time_bin_col]:
            # best-effort: if they exist but in different names, user should fix upstream
            logger.warning(f"Unexpected MultiIndex names: {idx_names}; expected ['patient_id', {time_bin_col!r}]")

    else:
        raise ValueError(f"Unknown output_schema.storage.index_representation={storage_mode!r}")

    # ----- 3) Infer feature/mask columns (manifest) -----
    feat_prefixes = (output_schema.get("features", {}) or {}).get("prefixes", [])
    mask_prefixes = (output_schema.get("masks", {}) or {}).get("prefixes", [])

    feature_cols, mask_cols = _infer_features_and_masks(
        X_out,
        feature_prefixes=feat_prefixes,
        mask_prefixes=mask_prefixes,
    )

    # ----- 4) Write dataset -----
    out_format = (cfg.get("output", {}) or {}).get("format", "parquet").lower()
    if out_format == "parquet":
        X_out.to_parquet(dataset_path, index=False if storage_mode == "columns" else True)
    elif out_format == "csv":
        X_out.to_csv(dataset_path, index=False if storage_mode == "columns" else True)
    else:
        raise ValueError(f"Unsupported output.format={out_format!r}; expected parquet|csv")

    # ----- 5) Write schema + features + meta -----
    schema_doc: Dict[str, Any] = {
        "schema_version": (cfg.get("metadata", {}) or {}).get("schema_version", 2),
        "storage": {"index_representation": storage_mode},
        "id_col": id_col,
        "time_col": time_col,
        "feature_selection": {
            "selection": "by_prefix",
            "prefixes": list(feat_prefixes),
        },
        "mask_selection": {
            "selection": "by_prefix",
            "prefixes": list(mask_prefixes),
        },
        "resolved_columns": {
            "feature_cols": feature_cols,
            "mask_cols": mask_cols,
        },
        "dtypes": {c: str(X_out[c].dtype) for c in X_out.columns} if storage_mode == "columns" else {},
    }
    _write_yaml(schema_path, schema_doc)

    _write_json(
        features_path,
        {
            "id_col": id_col,
            "time_col": time_col,
            "feature_cols": feature_cols,
            "mask_cols": mask_cols,
        },
    )

    now = datetime.now(timezone.utc).isoformat()

    meta: Dict[str, Any] = {
        "kind": "dataset",
        "dataset_id": dataset_id,
        "created_at": now,
        "inputs": {
            "raw_snapshot_id": raw_snapshot_id,
            "raw_snapshot_dir": str(raw_snapshot_dir),
            "raw_filename": input_filename,
        },
        "outputs": {
            "dataset_path": str(dataset_path),
            "schema_path": str(schema_path),
            "features_path": str(features_path),
            "n_rows": int(len(X_out)),
            "n_features": int(len(feature_cols)),
        },
        "config": {
            "config_path": config_path,
            "cfg_hash": cfg_hash,
        },
        "stage_params": {
            "history": cfg.get("history", {}),
            "filtering": cfg.get("filtering", {}),
            "featurization": cfg.get("featurization", {}),
            "time_bin_col": time_bin_col,
        },
        "featurization_params": featurizer.params_,
        "git_commit_hash": get_git_commit_hash(),
    }
    _write_yaml(meta_path, meta)

    _write_done(done_path)

    return DatasetArtifact(
        dataset_id=dataset_id,
        artifact_dir=str(artifact_dir),
        dataset_path=str(dataset_path),
        schema_path=str(schema_path),
        features_path=str(features_path),
        meta_path=str(meta_path),
        done_path=str(done_path),
    )
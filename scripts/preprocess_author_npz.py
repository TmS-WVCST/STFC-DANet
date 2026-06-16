import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert author-provided ABIDE CC400 NPZ files to local training cache."
    )
    parser.add_argument("--connectivity-npz", type=Path, required=True)
    parser.add_argument("--timeseries-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/abide_cc400_author_npz")
    parser.add_argument("--timeseries-key", default="fc")
    parser.add_argument("--connectivity-key", default="fc")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--subject-key", default="subject")
    parser.add_argument("--site-key", default="site")
    parser.add_argument("--demographic-keys", nargs="*", default=["AGE_AT_SCAN", "SEX"])
    parser.add_argument("--phenotypic-csv", type=Path, default=ROOT / "ABIDE/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    parser.add_argument("--phenotypic-id-key", default="FILE_ID")
    parser.add_argument("--target-timepoints", type=int, default=316)
    parser.add_argument("--crop-mode", choices=["first", "center"], default="first")
    parser.add_argument("--zscore-timeseries", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    print("{} keys:".format(path.name), list(data.keys()))
    for key in data.keys():
        value = data[key]
        print("  {}: shape={}, dtype={}".format(key, getattr(value, "shape", None), value.dtype))
    return {key: data[key] for key in data.keys()}


def get_key(data: dict, preferred: str, alternatives: list) -> str:
    if preferred in data:
        return preferred
    for key in alternatives:
        if key in data:
            return key
    raise KeyError("None of these keys were found: {}".format([preferred] + alternatives))


def normalize_subjects(subjects: np.ndarray) -> np.ndarray:
    return np.asarray([str(item) for item in subjects])


def align_by_subject(
    timeseries_data: dict,
    connectivity_data: dict,
    subject_key: str,
) -> tuple:
    ts_subject_key = get_key(timeseries_data, subject_key, ["subjects", "subject_ids", "sub_id"])
    conn_subject_key = get_key(connectivity_data, subject_key, ["subjects", "subject_ids", "sub_id"])
    ts_subjects = normalize_subjects(timeseries_data[ts_subject_key])
    conn_subjects = normalize_subjects(connectivity_data[conn_subject_key])

    ts_index = {subject: index for index, subject in enumerate(ts_subjects)}
    common_subjects = [subject for subject in conn_subjects if subject in ts_index]
    if not common_subjects:
        raise RuntimeError("No overlapping subject IDs found between NPZ files.")

    ts_indices = np.asarray([ts_index[subject] for subject in common_subjects], dtype=np.int64)
    conn_indices = np.asarray(
        [index for index, subject in enumerate(conn_subjects) if subject in ts_index],
        dtype=np.int64,
    )
    return common_subjects, ts_indices, conn_indices


def crop_timeseries(x: np.ndarray, target_timepoints: int, mode: str) -> np.ndarray:
    if x.shape[1] < target_timepoints:
        raise ValueError(
            "target_timepoints={} is larger than available T={}".format(
                target_timepoints,
                x.shape[1],
            )
        )
    if mode == "first":
        start = 0
    else:
        start = (x.shape[1] - target_timepoints) // 2
    return x[:, start : start + target_timepoints]


def zscore_per_subject(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-6)


def upper_triangle_from_connectivity(connectivity: np.ndarray) -> np.ndarray:
    if connectivity.ndim == 2:
        return connectivity.astype(np.float32)
    if connectivity.ndim != 3:
        raise ValueError("Connectivity array must be 2D or 3D, got shape {}".format(connectivity.shape))

    if connectivity.shape[1] == connectivity.shape[2]:
        matrices = connectivity
    else:
        matrices = []
        for sample in connectivity:
            corr = np.corrcoef(sample, rowvar=False)
            matrices.append(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
        matrices = np.stack(matrices, axis=0)

    indices = np.triu_indices(matrices.shape[1], k=1)
    return matrices[:, indices[0], indices[1]].astype(np.float32)


def make_domain(site_values: np.ndarray) -> tuple:
    site_strings = np.asarray([str(item) for item in site_values])
    unique_sites = sorted(set(site_strings))
    mapping = {site: index for index, site in enumerate(unique_sites)}
    domains = np.asarray([mapping[site] for site in site_strings], dtype=np.int64)
    return domains, mapping


def aligned_optional_metadata(
    key: str,
    timeseries_data: dict,
    connectivity_data: dict,
    ts_indices: np.ndarray,
    conn_indices: np.ndarray,
) -> np.ndarray:
    if key in connectivity_data and len(connectivity_data[key]) >= int(conn_indices.max()) + 1:
        return np.asarray(connectivity_data[key][conn_indices])
    if key in timeseries_data and len(timeseries_data[key]) >= int(ts_indices.max()) + 1:
        return np.asarray(timeseries_data[key][ts_indices])
    raise KeyError(key)


def phenotypic_metadata(
    phenotypic_csv: Path,
    phenotypic_id_key: str,
    subjects: list,
    demographic_keys: list,
) -> tuple:
    if not phenotypic_csv.exists():
        return {}, []
    phenotypic = pd.read_csv(phenotypic_csv)
    if phenotypic_id_key not in phenotypic.columns:
        return {}, demographic_keys

    selected = phenotypic[[phenotypic_id_key] + [key for key in demographic_keys if key in phenotypic.columns]]
    selected = selected.drop_duplicates(subset=[phenotypic_id_key])
    aligned = pd.DataFrame({"subject": subjects}).merge(
        selected,
        left_on="subject",
        right_on=phenotypic_id_key,
        how="left",
    )

    values = {}
    missing = []
    for key in demographic_keys:
        if key not in aligned.columns or aligned[key].isna().all():
            missing.append(key)
        else:
            values[key] = aligned[key].to_numpy()
    return values, missing


def main() -> None:
    args = parse_args()
    timeseries_data = load_npz(args.timeseries_npz)
    connectivity_data = load_npz(args.connectivity_npz)
    subjects, ts_indices, conn_indices = align_by_subject(
        timeseries_data,
        connectivity_data,
        args.subject_key,
    )

    ts_key = get_key(timeseries_data, args.timeseries_key, ["timeseries", "time_series", "X", "data"])
    conn_key = get_key(connectivity_data, args.connectivity_key, ["connectivity", "cc", "corr", "X", "data"])
    label_key = get_key(connectivity_data, args.label_key, ["labels", "y", "dx", "DX_GROUP"])
    site_key = get_key(connectivity_data, args.site_key, ["sites", "SITE_ID", "domain"])

    X = np.asarray(timeseries_data[ts_key][ts_indices], dtype=np.float32)
    if X.ndim != 3:
        raise ValueError("Time-series array must be [N, T, ROI], got {}".format(X.shape))
    X = crop_timeseries(X, args.target_timepoints, args.crop_mode)
    if args.zscore_timeseries:
        X = zscore_per_subject(X).astype(np.float32)

    connectivity = np.asarray(connectivity_data[conn_key][conn_indices])
    fc = upper_triangle_from_connectivity(connectivity)

    labels = np.asarray(connectivity_data[label_key][conn_indices]).astype(np.int64)
    labels = labels - labels.min()
    domains, site_mapping = make_domain(connectivity_data[site_key][conn_indices])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "X.npy", X)
    np.save(args.output_dir / "fc.npy", fc)
    np.save(args.output_dir / "y.npy", labels)
    np.save(args.output_dir / "domain.npy", domains)

    metadata_values = {
        "subject": subjects,
        "label": labels,
        "site": [str(item) for item in connectivity_data[site_key][conn_indices]],
        "domain": domains,
    }
    included_demographic_keys = []
    missing_demographic_keys = []
    pheno_values, pheno_missing = phenotypic_metadata(
        phenotypic_csv=args.phenotypic_csv,
        phenotypic_id_key=args.phenotypic_id_key,
        subjects=subjects,
        demographic_keys=args.demographic_keys,
    )
    for key in args.demographic_keys:
        if key in pheno_values:
            metadata_values[key] = pheno_values[key]
            included_demographic_keys.append(key)
            continue
        try:
            values = aligned_optional_metadata(
                key=key,
                timeseries_data=timeseries_data,
                connectivity_data=connectivity_data,
                ts_indices=ts_indices,
                conn_indices=conn_indices,
            )
            if len(values) == len(subjects):
                metadata_values[key] = values
                included_demographic_keys.append(key)
            else:
                missing_demographic_keys.append(key)
        except KeyError:
            missing_demographic_keys.append(key)
    missing_demographic_keys = sorted(set(missing_demographic_keys + pheno_missing) - set(included_demographic_keys))

    metadata = pd.DataFrame(metadata_values)
    metadata.to_csv(args.output_dir / "metadata.csv", index=False)
    with (args.output_dir / "site_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(site_mapping, f, indent=2)

    summary = {
        "num_samples": int(len(labels)),
        "num_asd": int((labels == 1).sum()),
        "num_control": int((labels == 0).sum()),
        "input_shape": list(X.shape),
        "fc_shape": list(fc.shape),
        "num_domains": int(len(site_mapping)),
        "connectivity_npz": str(args.connectivity_npz),
        "timeseries_npz": str(args.timeseries_npz),
        "timeseries_key": ts_key,
        "connectivity_key": conn_key,
        "label_key": label_key,
        "site_key": site_key,
        "included_demographic_keys": included_demographic_keys,
        "missing_demographic_keys": missing_demographic_keys,
        "phenotypic_csv": str(args.phenotypic_csv),
        "phenotypic_id_key": args.phenotypic_id_key,
    }
    with (args.output_dir / "preprocess_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Preprocessing completed.")
    print(summary)


if __name__ == "__main__":
    main()


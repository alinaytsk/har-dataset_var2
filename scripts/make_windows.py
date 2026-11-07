#!/usr/bin/env python3
"""Крок 3/3: нарізка на вікна, мапінг класів, розділення за людьми.

Гарантії:
  * вікно ніколи не перетинає розрив у даних, межу запису, активності чи учасника —
    бо ріжеться тільки в межах одного неперервного сегмента з data/interim;
  * неповне останнє вікно відкидається (drop_incomplete);
  * розділення за людьми: учасник потрапляє рівно в одну вибірку, інакше скрипт падає;
  * нормалізація у .npz НЕ застосовується — дані лишаються у фізичних одиницях
    (g і °/с), а mean/std по train зберігаються окремо в norm_stats.json;
  * файли пишуться детерміновано (фіксовані мітки часу в zip), тому повторний
    запуск дає байт-у-байт ідентичні .npz.

Вихід у data/processed/:
    train.npz, val.npz, test.npz   X float32 [N, windows.size_samples, len(channels)],
                                   y int64, y_raw int64, subject int64
                                   (+ метадані: channels, class_names, fs,
                                   window_samples, stride_samples, units)
    norm_stats.json                mean/std по 6 каналах ТІЛЬКИ з train
    label_map.json                 відповідність y, y_raw і класів RealWorld

Приклад:
    python scripts/make_windows.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numpy.lib import format as npy_format

ROOT = Path(__file__).resolve().parents[1]
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)       # фіксована мітка часу -> детерміновані файли


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def savez_deterministic(path: Path, **arrays: np.ndarray) -> None:
    """np.savez_compressed, але без поточного часу в zip-заголовках."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            npy_format.write_array(buf, np.asanyarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zf.writestr(info, buf.getvalue())


def build_label_maps(cfg: dict) -> tuple[dict[str, int], dict[str, int], list[str]]:
    """raw-активність -> y, raw-активність -> y_raw, назви цільових класів."""
    raw_classes = cfg["labels"]["raw_classes"]
    y_raw_of = {name: i for i, name in enumerate(raw_classes)}
    y_of: dict[str, int] = {}
    names: list[str] = []
    for spec in sorted(cfg["labels"]["classes"], key=lambda c: c["id"]):
        if spec["id"] != len(names):
            raise SystemExit("labels.classes: id мають іти підряд від 0")
        names.append(spec["name"])
        for raw in spec["raw"]:
            if raw not in y_raw_of:
                raise SystemExit(f"labels: активність {raw!r} немає в raw_classes")
            y_of[raw] = spec["id"]
    return y_of, y_raw_of, names


def check_split(cfg: dict) -> dict[int, str]:
    """subject -> назва вибірки; падає на перетині (це витік даних)."""
    owner: dict[int, str] = {}
    for name in ("train", "val", "test"):
        for subject in cfg["split"][name]:
            if subject in owner:
                raise SystemExit(f"учасник {subject} є і в {owner[subject]}, і в {name} — "
                                 "виправ split у config.yaml")
            owner[subject] = name
    return owner


def window_view(X: np.ndarray, size: int, stride: int) -> np.ndarray:
    """(T, C) -> (n, size, C); неповний хвіст відкидається."""
    n = (X.shape[0] - size) // stride + 1
    if n <= 0:
        return np.empty((0, size, X.shape[1]), dtype=X.dtype)
    s0, s1 = X.strides
    return np.lib.stride_tricks.as_strided(
        X, shape=(n, size, X.shape[1]), strides=(stride * s0, s0, s1), writeable=False)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="не перемішувати train (типово перемішується з seed з конфігу)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    interim = ROOT / cfg["paths"]["interim"]
    out_dir = ROOT / cfg["paths"]["processed"]
    manifest_path = interim / "manifest.csv"
    if not manifest_path.exists():
        print(f"Немає {manifest_path}. Спочатку: python scripts/preprocess.py",
              file=sys.stderr)
        return 1

    y_of, y_raw_of, class_names = build_label_maps(cfg)
    owner = check_split(cfg)
    size = int(cfg["windows"]["size_samples"])
    stride = int(cfg["windows"]["stride_samples"])
    channels = cfg["channels"]
    fs = float(cfg["processing"]["target_hz"])
    units = {"acc": cfg["dataset"]["sensors"]["acc"]["out_unit"],
             "gyr": cfg["dataset"]["sensors"]["gyr"]["out_unit"]}

    print(f"вікно {size} відліків ({size / fs:.2f} с), крок {stride} "
          f"(перекриття {100 * (1 - stride / size):.0f} %), {fs:g} Гц")
    print(f"класи: {dict(enumerate(class_names))}")
    print(f"відкинуті активності: {cfg['labels']['dropped']}")
    print(f"train={cfg['split']['train']} val={cfg['split']['val']} "
          f"test={cfg['split']['test']}\n")

    rows = pd.read_csv(manifest_path).sort_values(
        ["subject", "activity", "part", "segment"], kind="stable")

    buckets: dict[str, dict[str, list]] = {
        name: {"X": [], "y": [], "y_raw": [], "subject": []}
        for name in ("train", "val", "test")}
    unknown_subjects: set[int] = set()
    skipped_short = 0

    for _, row in rows.iterrows():
        subject, activity = int(row["subject"]), str(row["activity"])
        if activity in cfg["labels"]["dropped"] or activity not in y_of:
            continue
        split_name = owner.get(subject)
        if split_name is None:
            unknown_subjects.add(subject)
            continue

        with np.load(ROOT / row["file"], allow_pickle=False) as npz:
            X = npz["X"]
            seg_channels = [str(c) for c in npz["channels"]]
        if seg_channels != channels:
            print(f"! інші канали у {row['file']}: {seg_channels}", file=sys.stderr)
            return 1

        win = window_view(X, size, stride)
        if win.shape[0] == 0:
            skipped_short += 1
            continue
        b = buckets[split_name]
        b["X"].append(np.array(win, dtype="float32"))        # матеріалізуємо view
        b["y"].append(np.full(win.shape[0], y_of[activity], dtype="int64"))
        b["y_raw"].append(np.full(win.shape[0], y_raw_of[activity], dtype="int64"))
        b["subject"].append(np.full(win.shape[0], subject, dtype="int64"))

    if unknown_subjects:
        print(f"! учасники не згадані в split: {sorted(unknown_subjects)} — "
              "їх вікна не потрапили в жодну вибірку\n", file=sys.stderr)
    if skipped_short:
        print(f"сегментів, коротших за вікно: {skipped_short}")

    data: dict[str, dict[str, np.ndarray]] = {}
    for name, b in buckets.items():
        if not b["X"]:
            print(f"! вибірка {name} порожня", file=sys.stderr)
            continue
        data[name] = {k: np.concatenate(v) for k, v in b.items()}
    if "train" not in data:
        print("train порожній — зупиняюсь", file=sys.stderr)
        return 1

    if not args.no_shuffle:                       # детерміновано: seed з конфігу
        rng = np.random.default_rng(int(cfg["seed"]))
        order = rng.permutation(data["train"]["X"].shape[0])
        data["train"] = {k: v[order] for k, v in data["train"].items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(channels=np.array(channels), class_names=np.array(class_names),
                raw_class_names=np.array(cfg["labels"]["raw_classes"]),
                fs=np.float64(fs), window_samples=np.int64(size),
                stride_samples=np.int64(stride),
                units=np.array([units["acc"], units["gyr"]]))

    for name in ("train", "val", "test"):
        if name not in data:
            continue
        d = data[name]
        savez_deterministic(out_dir / f"{name}.npz", X=d["X"], y=d["y"],
                            y_raw=d["y_raw"], subject=d["subject"], **meta)
        counts = np.bincount(d["y"], minlength=len(class_names))
        print(f"{name:5s}: X{d['X'].shape} {d['X'].dtype} -> {name}.npz")
        for i, cls in enumerate(class_names):
            share = 100 * counts[i] / d["X"].shape[0] if d["X"].shape[0] else 0
            print(f"       {i} {cls:11s} {counts[i]:7d}  {share:5.1f}%")

    # norm_stats.json — ТІЛЬКИ з train.
    train_X = data["train"]["X"]
    flat = train_X.reshape(-1, train_X.shape[-1])
    mean, std = flat.mean(axis=0), flat.std(axis=0)
    (out_dir / "norm_stats.json").write_text(json.dumps({
        "computed_on": cfg["normalization"]["stats_from"],
        "subjects": sorted(int(s) for s in set(data["train"]["subject"].tolist())),
        "n_windows": int(train_X.shape[0]),
        "channels": channels,
        "units": units,
        "mean": [round(float(v), 8) for v in mean],
        "std": [round(float(v), 8) for v in std],
        "note": "Дані у .npz НЕ нормалізовані. Застосовувати як (X - mean) / std "
                "до train, val, test і до потоку з датчика на мікроконтролері.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    (out_dir / "label_map.json").write_text(json.dumps({
        "y": {str(spec["id"]): spec["name"]
              for spec in sorted(cfg["labels"]["classes"], key=lambda c: c["id"])},
        "y_to_raw_activities": {str(spec["id"]): spec["raw"]
                                for spec in sorted(cfg["labels"]["classes"],
                                                   key=lambda c: c["id"])},
        "y_raw": {str(i): name for i, name in enumerate(cfg["labels"]["raw_classes"])},
        "dropped_activities": cfg["labels"]["dropped"],
        "position": cfg["dataset"]["position"],
        "channels": channels,
        "units": units,
        "fs_hz": fs,
        "window_samples": size,
        "stride_samples": stride,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nnorm_stats.json: mean={[round(float(v), 4) for v in mean]}")
    print(f"                 std={[round(float(v), 4) for v in std]}")
    print(f"готово: {out_dir}")
    print("Далі: python scripts/checks.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

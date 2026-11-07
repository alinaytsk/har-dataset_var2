#!/usr/bin/env python3
"""Крок 2/3: очищення, синхронізація, одиниці, ресемплінг до processing.target_hz.

Що робиться з кожним записом (усі параметри — з config.yaml):
  1. читається csv (`id, attr_time, attr_x, attr_y, attr_z`, час — epoch у мс);
  2. одиниці переводяться у фізичні: acc м/с² -> g (÷9.80665), gyro рад/с -> °/с (×180/π);
  3. час чиститься: мс -> с, сортування, зняття дублікатів мітки;
  4. рахується ФАКТИЧНА частота за мітками (RealWorld декларує 50 Гц, реально 49.6-50.8);
  5. acc і gyro синхронізуються: беремо тільки спільний часовий інтервал;
  6. з кожного краю зрізається trim_seconds;
  7. розриви в мітках > max_gap_seconds не інтерполюються — запис ріжеться на
     неперервні сегменти (вікно потім ніколи не перетне розрив);
  8. сегмент інтерполюється на рівномірну сітку source_grid_hz (прибирає джитер),
     далі scipy.signal.resample_poly(up, down) -> target_hz з ФНЧ-антиаліасингом;
  9. сегмент зберігається в data/interim/ як .npz (X: (T, 6) float32, канали ax..gz).

Кожен файл активності (у т.ч. частина `_2`, `_3`) — окремий запис; вікна не
перетинають межу запису, активності чи учасника.

Приклади:
    python scripts/preprocess.py
    python scripts/preprocess.py --subjects 1,2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, filtfilt, resample_poly

ROOT = Path(__file__).resolve().parents[1]
SUBJECT_RE = re.compile(r"(?:^|[\\/])(?:s|proband)[_-]?(\d+)(?:[\\/]|$)", re.IGNORECASE)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_fname_re(position: str) -> re.Pattern:
    """acc_walking_upperarm.csv / Gyroscope_walking_2_upperarm.csv."""
    return re.compile(
        r"^(?P<sensor>[A-Za-z]+)_(?P<activity>[a-z]+)(?:_(?P<part>\d+))?_"
        rf"(?P<pos>{re.escape(position)})\.csv$", re.IGNORECASE)


def canon_sensor(raw: str, sensors: dict) -> str | None:
    raw = raw.lower()
    for key, spec in sensors.items():
        if raw in [a.lower() for a in spec["file_aliases"]]:
            return key
    return None


def scan_raw(raw_dir: Path, cfg: dict, subjects: list[int] | None):
    """{(subject, activity, part): {sensor: path}} + список нерозпізнаних файлів."""
    ds = cfg["dataset"]
    fname_re = build_fname_re(ds["position"])
    groups: dict[tuple, dict[str, Path]] = defaultdict(dict)
    unknown: list[str] = []

    for path in sorted(raw_dir.rglob("*.csv")):
        m = fname_re.match(path.name)
        sm = SUBJECT_RE.search(str(path.parent) + "/")
        if not m or not sm:
            unknown.append(path.name)
            continue
        sensor = canon_sensor(m.group("sensor"), ds["sensors"])
        activity = m.group("activity").lower()
        if sensor is None or activity not in ds["activities_used"]:
            continue
        subject = int(sm.group(1))
        if subjects is not None and subject not in subjects:
            continue
        groups[(subject, activity, int(m.group("part") or 1))][sensor] = path

    return dict(groups), unknown


def read_sensor_csv(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(t у секундах від початку, xyz (N,3) у цільових одиницях, фактична частота)."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*keys: str) -> str:
        for key in keys:
            for low, orig in cols.items():
                if low in (key, f"attr_{key}") or low.endswith(f"_{key}"):
                    return orig
        raise KeyError(f"{path.name}: немає колонки {keys}; є {list(df.columns)}")

    t = pd.to_numeric(df[pick("time", "timestamp")], errors="coerce").to_numpy("float64")
    xyz = (df[[pick("x"), pick("y"), pick("z")]]
           .apply(pd.to_numeric, errors="coerce").to_numpy("float64"))

    good = np.isfinite(t) & np.isfinite(xyz).all(axis=1)
    t, xyz = t[good], xyz[good]
    if t.size < 2:
        return t, xyz, float("nan")

    if np.nanmedian(np.abs(t)) > 1e11:      # epoch у мілісекундах
        t = t / 1000.0
    order = np.argsort(t, kind="stable")
    t, xyz = t[order], xyz[order]
    keep = np.concatenate(([True], np.diff(t) > 0))     # дублікати мітки часу
    t, xyz = t[keep], xyz[keep]

    measured_hz = (t.size - 1) / (t[-1] - t[0]) if t[-1] > t[0] else float("nan")
    return t - t[0], xyz * scale, measured_hz


def gap_mask(src_t: np.ndarray, grid: np.ndarray, max_gap: float) -> np.ndarray:
    """True там, де точку сітки реально накривають сирі семпли."""
    ok = (grid >= src_t[0]) & (grid <= src_t[-1])
    for i in np.flatnonzero(np.diff(src_t) > max_gap):
        ok &= ~((grid > src_t[i]) & (grid < src_t[i + 1]))
    return ok


def to_grid(src_t: np.ndarray, src_x: np.ndarray, grid: np.ndarray, how: str) -> np.ndarray:
    if how == "nearest":
        idx = np.searchsorted(src_t, grid).clip(1, src_t.size - 1)
        idx = np.where((grid - src_t[idx - 1]) <= (src_t[idx] - grid), idx - 1, idx)
        return src_x[idx]
    return np.stack([np.interp(grid, src_t, src_x[:, c]) for c in range(src_x.shape[1])],
                    axis=1)


def antialias(src_t: np.ndarray, src_x: np.ndarray, measured_hz: float,
              grid_hz: float, cfg_aa: dict) -> tuple[np.ndarray, np.ndarray]:
    """ФНЧ для записів, що писалися швидше за цільову сітку.

    Без цього пониження (напр. 101 -> 50 Гц) простою інтерполяцією склало б
    усе вище 25 Гц у робочу смугу. Записи на ~50 Гц повертаються без змін.
    """
    if not cfg_aa["enabled"] or not np.isfinite(measured_hz):
        return src_t, src_x
    if measured_hz <= grid_hz * float(cfg_aa["faster_than_ratio"]):
        return src_t, src_x

    fine_hz = float(np.ceil(measured_hz))
    n = int(np.floor((src_t[-1] - src_t[0]) * fine_hz)) + 1
    fine_t = src_t[0] + np.arange(n) / fine_hz
    fine_x = np.stack([np.interp(fine_t, src_t, src_x[:, c])
                       for c in range(src_x.shape[1])], axis=1)

    order = int(cfg_aa["order"])
    cutoff = float(cfg_aa["cutoff_ratio"]) * grid_hz
    if fine_x.shape[0] <= 3 * (2 * order + 1):
        return src_t, src_x
    b, a = butter(order, cutoff / (fine_hz / 2.0), btype="low")
    return fine_t, filtfilt(b, a, fine_x, axis=0)


def contiguous_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def process_record(key: tuple, files: dict[str, Path], cfg: dict) -> tuple[list[dict], dict]:
    """Один запис -> (список сегментів на target_hz, діагностика)."""
    subject, activity, part = key
    ds, pr = cfg["dataset"], cfg["processing"]
    g = float(cfg["constants"]["g"])
    scales = {"acc": 1.0 / g, "gyr": 180.0 / math.pi}

    series, diag = {}, {"subject": subject, "activity": activity, "part": part}
    for sensor in ds["sensors"]:
        t, xyz, hz = read_sensor_csv(files[sensor], scales[sensor])
        if t.size < 2:
            diag["error"] = f"{sensor}: порожній або битий csv"
            return [], diag
        series[sensor] = (t, xyz)
        diag[f"{sensor}_hz"] = round(hz, 3)
        diag[f"{sensor}_samples"] = int(t.size)
        diag[f"{sensor}_duration_s"] = round(float(t[-1]), 2)

    # Синхронізація: спільний інтервал обох сенсорів + обрізання країв.
    t0 = max(t[0] for t, _ in series.values()) + float(pr["trim_seconds"])
    t1 = min(t[-1] for t, _ in series.values()) - float(pr["trim_seconds"])
    diag["overlap_s"] = round(max(0.0, t1 - t0), 2)
    if t1 - t0 < float(pr["min_segment_seconds"]):
        diag["error"] = "після синхронізації й обрізання коротше за мінімальний сегмент"
        return [], diag

    grid_hz = float(pr["source_grid_hz"])
    n = int(np.floor((t1 - t0) * grid_hz)) + 1
    grid = t0 + np.arange(n) / grid_hz

    valid = np.ones(n, dtype=bool)
    blocks = []
    for sensor in ds["sensors"]:                     # порядок каналів: ax..az, gx..gz
        src_t, src_x = series[sensor]
        # Маска розривів — по СИРИХ мітках, до будь-якої фільтрації.
        valid &= gap_mask(src_t, grid, float(pr["max_gap_seconds"]))
        filt_t, filt_x = antialias(src_t, src_x, diag.get(f"{sensor}_hz", float("nan")),
                                   grid_hz, pr["antialias"])
        diag[f"{sensor}_antialiased"] = filt_t is not src_t
        blocks.append(to_grid(filt_t, filt_x, grid, pr["interp"]))
    X = np.concatenate(blocks, axis=1)

    up, down = int(pr["resample_up"]), int(pr["resample_down"])
    min_len = int(round(float(pr["min_segment_seconds"]) * grid_hz))
    runs = contiguous_runs(valid, min_len)
    diag["invalid_samples"] = int((~valid).sum())
    diag["segments"] = len(runs)

    out = []
    for i, (start, stop) in enumerate(runs):
        seg = resample_poly(X[start:stop], up, down, axis=0)   # сітка -> target_hz з ФНЧ
        out.append({"subject": subject, "activity": activity, "part": part,
                    "segment": i, "X": np.ascontiguousarray(seg, dtype="float32")})
    return out, diag


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--subjects", help="наприклад 1,2 (типово — усі знайдені)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds, pr = cfg["dataset"], cfg["processing"]
    raw_dir, interim = ROOT / cfg["paths"]["raw"], ROOT / cfg["paths"]["interim"]
    if not raw_dir.exists():
        print(f"Немає {raw_dir}. Спочатку: python scripts/download.py", file=sys.stderr)
        return 1

    subjects = None
    if args.subjects:
        subjects = [int(x) for x in args.subjects.replace(" ", "").split(",") if x]

    fs_target = float(pr["target_hz"])
    print(f"сирі дані: {raw_dir}")
    print(f"позиція: {ds['position']}, сенсори: {list(ds['sensors'])}")
    print(f"одиниці: acc -> {ds['sensors']['acc']['out_unit']}, "
          f"gyro -> {ds['sensors']['gyr']['out_unit']}")
    print(f"ресемплінг: {pr['source_grid_hz']} -> {fs_target} Гц "
          f"(resample_poly up={pr['resample_up']} down={pr['resample_down']})\n")

    groups, unknown = scan_raw(raw_dir, cfg, subjects)
    print(f"знайдено записів: {len(groups)}"
          + (f", нерозпізнаних файлів: {len(unknown)}" if unknown else ""))

    rows: list[dict] = []
    diags: list[dict] = []
    incomplete: list[str] = []
    channels = cfg["channels"]

    for key in sorted(groups):
        subject, activity, part = key
        files = groups[key]
        missing = [s for s in ds["sensors"] if s not in files]
        if missing:
            incomplete.append(f"s{subject}/{activity}_p{part}: немає {missing}")
            continue

        segments, diag = process_record(key, files, cfg)
        diags.append(diag)
        if not segments:
            print(f"  s{subject} {activity} p{part}: {diag.get('error', 'немає сегментів')}")
            continue

        for seg in segments:
            out_dir = interim / f"s{subject}"
            out_dir.mkdir(parents=True, exist_ok=True)
            name = f"{activity}_p{part}_seg{seg['segment']}.npz"
            np.savez_compressed(out_dir / name, X=seg["X"],
                                channels=np.array(channels), fs=fs_target,
                                subject=subject, activity=activity,
                                part=part, segment=seg["segment"])
            acc_mag = float(np.linalg.norm(seg["X"][:, :3], axis=1).mean())
            rows.append({
                "file": (out_dir / name).relative_to(ROOT).as_posix(),
                "subject": subject, "activity": activity, "part": part,
                "segment": seg["segment"], "n_samples": seg["X"].shape[0],
                "duration_s": round(seg["X"].shape[0] / fs_target, 3),
                "fs": fs_target,
                "acc_hz_raw": diag.get("acc_hz"), "gyr_hz_raw": diag.get("gyr_hz"),
                "acc_mag_mean_g": round(acc_mag, 4),
            })
        total_s = sum(s["X"].shape[0] for s in segments) / fs_target
        print(f"  s{subject} {activity} p{part}: {len(segments)} сегм., {total_s:7.1f} с, "
              f"сира частота acc={diag.get('acc_hz')} gyr={diag.get('gyr_hz')} Гц")

    if not rows:
        print("\nНічого не оброблено — перевір вміст data/raw.", file=sys.stderr)
        return 1

    interim.mkdir(parents=True, exist_ok=True)
    manifest = interim / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Файли від попереднього прогону, яких уже немає в manifest: не видаляємо
    # (make_windows.py читає лише manifest), але повідомляємо.
    listed = {ROOT / r["file"] for r in rows}
    stale = sorted(p.relative_to(ROOT).as_posix()
                   for p in interim.rglob("*.npz") if p not in listed)

    # Повнота: у кожного учасника кожна використана активність.
    have = {(r["subject"], r["activity"]) for r in rows}
    want_subjects = subjects or ds["subjects"]
    gaps = [f"s{s}/{a}" for s in want_subjects for a in ds["activities_used"]
            if (s, a) not in have]

    hz = [d[f"{s}_hz"] for d in diags for s in ds["sensors"] if f"{s}_hz" in d]
    tol = float(pr["freq_tolerance_hz"])
    exp = float(pr["expected_source_hz"])
    outliers = [round(h, 2) for h in hz if not (exp - tol <= h <= exp + tol)]

    report = {
        "records": len(diags), "segments": len(rows),
        "hours": round(sum(r["duration_s"] for r in rows) / 3600, 3),
        "raw_frequency_hz": {
            "min": round(min(hz), 3), "max": round(max(hz), 3),
            "mean": round(float(np.mean(hz)), 3),
            "outside_tolerance": outliers,
        },
        "antialiased_records": [f"s{d['subject']}/{d['activity']}_p{d['part']}:{s}"
                                for d in diags for s in ds["sensors"]
                                if d.get(f"{s}_antialiased")],
        "missing_subject_activity": gaps,
        "incomplete_records": incomplete,
        "stale_interim_files": stale,
        "unknown_files": unknown[:50],
        "per_record": diags,
    }
    (interim / "preprocess_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nсегментів: {len(rows)}, сумарно {report['hours']:.2f} год")
    print(f"фактична частота сирих даних: {report['raw_frequency_hz']['min']}–"
          f"{report['raw_frequency_hz']['max']} Гц "
          f"(середня {report['raw_frequency_hz']['mean']})")
    if outliers:
        print(f"! поза допуском ±{tol} Гц: {outliers} — застосовано антиаліасинговий ФНЧ")
    if report["antialiased_records"]:
        print(f"  префільтровано записів: {report['antialiased_records']}")
    if stale:
        print(f"! у {interim} лежить {len(stale)} .npz від попереднього прогону "
              "(не використовуються; можна видалити вручну)")
    if incomplete:
        print(f"! записів без пари acc+gyr: {len(incomplete)}")
        for line in incomplete:
            print(f"    {line}")
    if gaps:
        print(f"! немає даних для {len(gaps)} пар (учасник/активність):")
        for line in gaps:
            print(f"    {line}")
    print(f"manifest: {manifest}")
    print("Далі: python scripts/make_windows.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

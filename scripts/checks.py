#!/usr/bin/env python3
"""Крок 4/3 :) — перевірки якості датасету і графіки у reports/.

Перевіряє все, що має бути перевірено перед передачею датасету:
  1. форми і типи масивів відповідають config.yaml;
  2. немає NaN/inf;
  3. учасники не перетинаються між train/val/test;
  4. у кожного учасника є потрібна позиція для кожної активності;
  5. у val і test присутні всі цільові класи;
  6. кількість вікон по класах у кожній вибірці;
  7. |acc| для статичних класів ≈ 1 g (перевірка одиниць);
  8. діапазони значень по каналах у межах фізично можливого;
  9. повторний запуск дає ідентичні файли (--verify-rerun).

Пише reports/checks.md, reports/checks.json, reports/file_hashes.json і графіки.

Приклади:
    python scripts/checks.py
    python scripts/checks.py --verify-rerun     # перезапускає крок 3 і порівнює хеші
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
PNG_META = {"Software": None}


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_splits(proc: Path) -> dict[str, dict]:
    out = {}
    for name in SPLITS:
        path = proc / f"{name}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as npz:
                out[name] = {k: npz[k] for k in npz.files}
    return out


def plot_class_distribution(data: dict, class_names: list[str], out: Path) -> None:
    counts = {name: np.bincount(d["y"], minlength=len(class_names))
              for name, d in data.items()}
    x = np.arange(len(class_names))
    width = 0.8 / max(1, len(counts))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, (name, c) in enumerate(counts.items()):
        ax1.bar(x + i * width - 0.4 + width / 2, c, width, label=name)
        total = c.sum() or 1
        ax2.bar(x + i * width - 0.4 + width / 2, 100 * c / total, width, label=name)
    for ax, ylab, title in ((ax1, "вікон", "Кількість вікон по класах"),
                            (ax2, "%", "Частка класу у вибірці")):
        ax.set_xticks(x, class_names, rotation=20, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130, metadata=PNG_META)
    plt.close(fig)


def plot_class_windows(data: dict, class_names: list[str], channels: list[str],
                       fs: float, units: dict, out_dir: Path, seed: int) -> list[str]:
    """По одному графіку на клас: 2 приклади вікон × 6 каналів."""
    rng = np.random.default_rng(seed)
    d = data["train"]
    t = np.arange(d["X"].shape[1]) / fs
    written = []
    for cls_id, cls in enumerate(class_names):
        idx = np.flatnonzero(d["y"] == cls_id)
        if idx.size == 0:
            continue
        picks = rng.choice(idx, size=min(2, idx.size), replace=False)
        fig, axes = plt.subplots(2, len(picks), figsize=(6 * len(picks), 6),
                                 squeeze=False, sharex=True)
        for col, w in enumerate(picks):
            win = d["X"][w]
            subject = int(d["subject"][w])
            for ch in range(3):
                axes[0][col].plot(t, win[:, ch], lw=1.1, label=channels[ch])
                axes[1][col].plot(t, win[:, 3 + ch], lw=1.1, label=channels[3 + ch])
            axes[0][col].set_title(f"{cls}, учасник s{subject}")
            axes[0][col].set_ylabel(f"акселерометр, {units['acc']}")
            axes[1][col].set_ylabel(f"гіроскоп, {units['gyr']}")
            axes[1][col].set_xlabel("час, с")
            for ax in (axes[0][col], axes[1][col]):
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8, ncol=3, loc="upper right")
        fig.suptitle(f"Приклади вікон: клас {cls_id} — {cls}")
        fig.tight_layout()
        path = out_dir / f"windows_{cls_id}_{cls}.png"
        fig.savefig(path, dpi=120, metadata=PNG_META)
        plt.close(fig)
        written.append(path.name)
    return written


def plot_subject_matrix(data: dict, class_names: list[str], out: Path) -> None:
    subjects = sorted({int(s) for d in data.values() for s in d["subject"]})
    grid = np.zeros((len(subjects), len(class_names)), dtype=int)
    row_of = {s: i for i, s in enumerate(subjects)}
    for d in data.values():
        for subject, y in zip(d["subject"], d["y"]):
            grid[row_of[int(subject)], int(y)] += 1

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(subjects) + 2.5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(class_names)), class_names, rotation=20, ha="right")
    ax.set_yticks(range(len(subjects)), [f"s{s}" for s in subjects])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=7,
                    color="w" if grid[i, j] < grid.max() * 0.6 else "k")
    ax.set_title("Кількість вікон: учасник × клас")
    fig.colorbar(im, ax=ax, label="вікон")
    fig.tight_layout()
    fig.savefig(out, dpi=130, metadata=PNG_META)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--verify-rerun", action="store_true",
                    help="перезапустити make_windows.py і порівняти хеші файлів")
    args = ap.parse_args()

    cfg = load_config(args.config)
    proc = ROOT / cfg["paths"]["processed"]
    reports = ROOT / cfg["paths"]["reports"]
    reports.mkdir(parents=True, exist_ok=True)

    data = load_splits(proc)
    if "train" not in data:
        print(f"Немає {proc}/train.npz. Спочатку: python scripts/make_windows.py",
              file=sys.stderr)
        return 1

    class_names = [str(c) for c in data["train"]["class_names"]]
    channels = [str(c) for c in data["train"]["channels"]]
    units = {"acc": str(data["train"]["units"][0]), "gyr": str(data["train"]["units"][1])}
    fs = float(data["train"]["fs"])
    size = int(cfg["windows"]["size_samples"])
    checks: list[dict] = []
    results: dict = {"splits": {}, "checks": []}

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"{'OK  ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))

    # 1. Форми і типи.
    expect = {"X": ("float32", 3), "y": ("int64", 1),
              "y_raw": ("int64", 1), "subject": ("int64", 1)}
    bad_shapes = []
    for name, d in data.items():
        n = d["X"].shape[0]
        for key, (dtype, ndim) in expect.items():
            if d[key].dtype != dtype or d[key].ndim != ndim or d[key].shape[0] != n:
                bad_shapes.append(f"{name}.{key}: {d[key].dtype} {d[key].shape}")
        if d["X"].shape[1:] != (size, len(channels)):
            bad_shapes.append(f"{name}.X: {d['X'].shape}, очікувано (N,{size},{len(channels)})")
        results["splits"][name] = {
            "n_windows": int(n), "X_shape": list(d["X"].shape),
            "dtypes": {k: str(d[k].dtype) for k in expect},
            "subjects": sorted({int(s) for s in d["subject"]}),
            "hours_no_overlap": round(n * int(cfg["windows"]["stride_samples"]) / fs / 3600, 3),
        }
    check("форми і типи масивів відповідають специфікації", not bad_shapes,
          "; ".join(bad_shapes))

    # 2. NaN / inf.
    nan_counts = {name: int((~np.isfinite(d["X"])).sum()) for name, d in data.items()}
    check("немає NaN/inf", all(v == 0 for v in nan_counts.values()), str(nan_counts))

    # 3. Перетин учасників.
    subs = {name: {int(s) for s in d["subject"]} for name, d in data.items()}
    overlaps = [f"{a}∩{b}={sorted(subs[a] & subs[b])}"
                for a in subs for b in subs if a < b and subs[a] & subs[b]]
    check("учасники не перетинаються між вибірками", not overlaps, "; ".join(overlaps))
    results["subjects_per_split"] = {k: sorted(v) for k, v in subs.items()}

    # 4. Повнота сирих даних (позиція × активність × учасник).
    pre_report = ROOT / cfg["paths"]["interim"] / "preprocess_report.json"
    missing: list[str] = []
    if pre_report.exists():
        pre = json.loads(pre_report.read_text(encoding="utf-8"))
        missing = pre.get("missing_subject_activity", []) + pre.get("incomplete_records", [])
        results["raw_frequency_hz"] = pre.get("raw_frequency_hz")
        results["records"] = pre.get("records")
        results["segments"] = pre.get("segments")
    check(f"у кожного учасника є {cfg['dataset']['position']} для кожної активності",
          not missing, ", ".join(missing))

    # 5. Усі класи у val і test.
    absent = []
    for name in ("val", "test"):
        if name not in data:
            absent.append(f"{name}: вибірки немає")
            continue
        got = set(data[name]["y"].tolist())
        gap = [class_names[i] for i in range(len(class_names)) if i not in got]
        if gap:
            absent.append(f"{name}: немає {gap}")
    check("у val і test присутні всі класи", not absent, "; ".join(absent))

    # 6. Кількість вікон по класах.
    per_class = {}
    for name, d in data.items():
        counts = np.bincount(d["y"], minlength=len(class_names))
        per_class[name] = {class_names[i]: int(counts[i]) for i in range(len(class_names))}
        results["splits"][name]["per_class"] = per_class[name]
    check("кількість вікон по класах порахована", True,
          f"train={sum(per_class['train'].values())} вікон")

    # 7. |acc| ≈ 1 g для статичних класів.
    lo, hi = cfg["checks"]["static_acc_g_range"]
    static_rows, static_ok = {}, True
    for cls in cfg["checks"]["static_classes"]:
        cls_id = class_names.index(cls)
        for name, d in data.items():
            sel = d["X"][d["y"] == cls_id]
            if sel.shape[0] == 0:
                continue
            mag = float(np.linalg.norm(sel[:, :, :3], axis=2).mean())
            static_rows[f"{name}/{cls}"] = round(mag, 4)
            static_ok &= lo <= mag <= hi
    results["static_acc_magnitude_g"] = static_rows
    check(f"|acc| для статичних класів у межах {lo}–{hi} g", static_ok, str(static_rows))

    # 8. Діапазони по каналах.
    acc_lim, gyr_lim = cfg["checks"]["acc_limit_g"], cfg["checks"]["gyr_limit_dps"]
    ranges, range_ok = {}, True
    for name, d in data.items():
        per_ch = {}
        for i, ch in enumerate(channels):
            col = d["X"][:, :, i]
            lim = acc_lim if i < 3 else gyr_lim
            mn, mx = float(col.min()), float(col.max())
            per_ch[ch] = [round(mn, 3), round(mx, 3)]
            range_ok &= abs(mn) <= lim and abs(mx) <= lim
        ranges[name] = per_ch
    results["channel_ranges"] = ranges
    check(f"діапазони каналів у межах ±{acc_lim} g / ±{gyr_lim} °/с", range_ok)

    # 9. Детермінованість.
    hashes = {p.name: sha256(p) for p in sorted(proc.glob("*")) if p.is_file()}
    results["file_hashes"] = hashes
    hashes_path = reports / "file_hashes.json"
    if args.verify_rerun:
        print("\nперезапускаю make_windows.py для перевірки детермінованості...")
        run = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_windows.py")],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", cwd=ROOT)
        if run.returncode != 0:
            check("повторний запуск дає ідентичні файли", False,
                  f"make_windows.py впав: {run.stderr.strip()[-300:]}")
        else:
            again = {p.name: sha256(p) for p in sorted(proc.glob("*")) if p.is_file()}
            diff = [k for k in hashes if hashes.get(k) != again.get(k)]
            check("повторний запуск дає ідентичні файли", not diff,
                  f"відрізняються: {diff}" if diff else "усі хеші збігаються")
    elif hashes_path.exists():
        old = json.loads(hashes_path.read_text(encoding="utf-8"))
        same = [k for k in hashes if k in old and old[k] == hashes[k]]
        check("хеші збігаються з попереднім запуском", len(same) == len(hashes),
              f"{len(same)}/{len(hashes)} файлів без змін")
    else:
        print("INFO хеші збережено; для перевірки детермінованості: --verify-rerun")
    hashes_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    # ── Графіки ──────────────────────────────────────────────────────────────
    plot_class_distribution(data, class_names, reports / "class_distribution.png")
    plot_subject_matrix(data, class_names, reports / "subject_class_matrix.png")
    pngs = plot_class_windows(data, class_names, channels, fs, units, reports,
                              int(cfg["seed"]))
    print(f"\nграфіки: class_distribution.png, subject_class_matrix.png, "
          f"{len(pngs)} × windows_*.png")

    # ── Звіти ────────────────────────────────────────────────────────────────
    results["checks"] = checks
    results["config"] = {
        "position": cfg["dataset"]["position"], "channels": channels, "units": units,
        "fs_hz": fs, "window_samples": size,
        "stride_samples": int(cfg["windows"]["stride_samples"]),
        "classes": class_names, "dropped": cfg["labels"]["dropped"],
        "split": cfg["split"], "seed": cfg["seed"],
    }
    (reports / "checks.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Результати перевірок датасету", "",
          f"Позиція: `{cfg['dataset']['position']}` · канали: `{', '.join(channels)}` · "
          f"одиниці: acc `{units['acc']}`, gyro `{units['gyr']}` · частота: {fs:g} Гц · "
          f"вікно: {size} × {len(channels)} (крок {cfg['windows']['stride_samples']})", "",
          "## Перевірки", "", "| Перевірка | Результат | Деталі |", "|---|---|---|"]
    for c in checks:
        detail = (c["detail"] or "").replace("|", "\\|")
        md.append(f"| {c['name']} | {'OK' if c['ok'] else 'FAIL'} | {detail[:400]} |")

    md += ["", "## Вибірки", "",
           "| Вибірка | Вікон | Форма X | Учасники | Годин (без перекриття) |",
           "|---|---|---|---|---|"]
    for name in SPLITS:
        if name not in results["splits"]:
            continue
        s = results["splits"][name]
        md.append(f"| {name} | {s['n_windows']} | {tuple(s['X_shape'])} | "
                  f"{s['subjects']} | {s['hours_no_overlap']} |")

    md += ["", "## Кількість вікон по класах", "",
           "| Клас | " + " | ".join(n for n in SPLITS if n in per_class) + " |",
           "|---" * (1 + len([n for n in SPLITS if n in per_class])) + "|"]
    for cls in class_names:
        md.append(f"| {cls} | " + " | ".join(str(per_class[n][cls])
                                             for n in SPLITS if n in per_class) + " |")

    md += ["", "## |acc| для статичних класів (очікується ≈ 1 g)", "",
           "| Вибірка / клас | середній \\|acc\\|, g |", "|---|---|"]
    for key, value in static_rows.items():
        md.append(f"| {key} | {value} |")

    md += ["", "## Діапазони значень по каналах", "",
           "| Вибірка | " + " | ".join(channels) + " |",
           "|---" * (1 + len(channels)) + "|"]
    for name in SPLITS:
        if name not in ranges:
            continue
        md.append(f"| {name} | " + " | ".join(
            f"{ranges[name][ch][0]} … {ranges[name][ch][1]}" for ch in channels) + " |")

    if results.get("raw_frequency_hz"):
        rf = results["raw_frequency_hz"]
        md += ["", "## Фактична частота сирих даних", "",
               f"Виміряно за мітками часу: {rf['min']}–{rf['max']} Гц "
               f"(середня {rf['mean']}). Заявлено на сайті: "
               f"{cfg['processing']['expected_source_hz']} Гц.", ""]

    md += ["", "## Графіки", "",
           "* `class_distribution.png` — кількість і частка вікон по класах у вибірках",
           "* `subject_class_matrix.png` — вікна в розрізі учасник × клас",
           *[f"* `{p}` — приклади вікон (6 каналів)" for p in pngs], ""]

    (reports / "checks.md").write_text("\n".join(md), encoding="utf-8")
    print(f"звіти: {reports / 'checks.md'}, {reports / 'checks.json'}")

    failed = [c["name"] for c in checks if not c["ok"]]
    if failed:
        print(f"\nПРОВАЛЕНО перевірок: {len(failed)}", file=sys.stderr)
        for name in failed:
            print(f"  - {name}", file=sys.stderr)
        return 1
    print("\nУсі перевірки пройдено.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

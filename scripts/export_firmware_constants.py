#!/usr/bin/env python3
"""Генерує C-заголовок для прошивки з config.yaml і norm_stats.json.

Сенс: частота, розмір вікна, порядок каналів, одиниці, статистики нормалізації
і коефіцієнти ФНЧ мають бути ОДНАКОВІ в датасеті і на платі. Тримати їх у двох
місцях вручну — гарантовано розійтися, тому заголовок генерується.

Коефіцієнти ФНЧ рахує scipy (той самий інструмент, що й у препроцесингу),
у вигляді SOS — каскаду біквадів, стабільного у float32.

Приклади:
    python scripts/export_firmware_constants.py            # у шлях із config.yaml
    python scripts/export_firmware_constants.py --out x.h  # у довільний файл
    python scripts/export_firmware_constants.py --print    # у stdout, нічого не пише
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def c_float(x: float) -> str:
    return f"{float(x):+.9e}f"


def build_header(cfg: dict, stats: dict, guard: str) -> str:
    fs = float(cfg["processing"]["target_hz"])
    size = int(cfg["windows"]["size_samples"])
    hop = int(cfg["windows"]["stride_samples"])
    channels = cfg["channels"]
    fw = cfg["firmware"]
    cutoff = float(fw["lowpass_hz"])
    order = int(fw["lowpass_order"])
    # Той самий поріг розриву, що й у препроцесингу датасету: менші провали
    # добиваються інтерполяцією, більші рвуть вікно.
    max_gap_s = float(cfg["processing"]["max_gap_seconds"])
    max_gap = int(round(max_gap_s * fs))

    nyq = fs / 2.0
    if not 0.0 < cutoff < nyq:
        raise SystemExit(f"firmware.lowpass_hz={cutoff} має бути в (0, {nyq})")
    sos = butter(order, cutoff / nyq, btype="low", output="sos")

    class_names = [c["name"] for c in sorted(cfg["labels"]["classes"],
                                             key=lambda c: c["id"])]
    units = {"acc": cfg["dataset"]["sensors"]["acc"]["out_unit"],
             "gyr": cfg["dataset"]["sensors"]["gyr"]["out_unit"]}

    lines = [
        "/* ЗГЕНЕРОВАНО АВТОМАТИЧНО — не редагувати руками.",
        " *",
        f" * Джерело: har-dataset/config.yaml + data/processed/norm_stats.json",
        f" * Перегенерувати: python scripts/export_firmware_constants.py",
        " *",
        " * Ці константи описують рівно той формат даних, на якому навчена модель:",
        f" * {fs:g} Гц, вікно {size} x {len(channels)}, канали "
        f"{', '.join(channels)}, acc у {units['acc']}, gyro у {units['gyr']}.",
        " */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "/* ── Формат входу моделі ─────────────────────────────────────────── */",
        f"#define HAR_FS_HZ            {fs:.1f}f",
        f"#define HAR_CHANNELS         {len(channels)}",
        f"#define HAR_WINDOW_SAMPLES   {size}      /* {size / fs:.2f} с */",
        f"#define HAR_HOP_SAMPLES      {hop}      /* перекриття "
        f"{100 * (1 - hop / size):.0f} % */",
        f"#define HAR_MAX_GAP_SAMPLES  {max_gap}      /* {max_gap_s:g} с: "
        f"більший розрив рве вікно */",
        f"#define HAR_ACC_UNIT         \"{units['acc']}\"",
        f"#define HAR_GYR_UNIT         \"{units['gyr']}\"",
        "",
        "/* Порядок каналів у вікні (рядок = момент часу): */",
        "/*   " + ", ".join(f"{i}:{c}" for i, c in enumerate(channels)) + " */",
        "",
        "/* ── Класи ───────────────────────────────────────────────────────── */",
        f"#define HAR_CLASS_COUNT      {len(class_names)}",
        "#define HAR_CLASS_LIST       "
        + ", ".join(f'"{n}"' for n in class_names),
        "",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"#define HAR_CLASS_{name.upper():<12} {i}")

    lines += [
        "",
        "/* ── ФНЧ перед інференсом ────────────────────────────────────────── */",
        "/* Баттерворт " + str(order) + f"-го порядку, зріз {cutoff:g} Гц при "
        f"{fs:g} Гц, у вигляді каскаду біквадів (SOS).",
        " * Потрібен обов'язково: навчальні дані отримані з 50-герцового джерела,",
        " * тому вище 25 Гц у них сигналу немає, а датчик на платі туди шумить. */",
        f"#define HAR_LPF_HZ           {cutoff:.1f}f",
        f"#define HAR_LPF_ORDER        {order}",
        f"#define HAR_LPF_SECTIONS     {sos.shape[0]}",
        "",
        "/* {b0, b1, b2, a1, a2} на секцію; a0 нормалізований до 1. */",
        "#define HAR_LPF_SOS { \\",
    ]
    for row in sos:
        b0, b1, b2, _, a1, a2 = row
        lines.append("    { " + ", ".join(c_float(v) for v in (b0, b1, b2, a1, a2))
                     + " }, \\")
    lines[-1] = lines[-1].rstrip(" \\").rstrip(",") + " \\"
    lines += ["}", ""]

    lines += [
        "/* ── Нормалізація (статистики з train, НЕ перераховувати) ────────── */",
        f"/* джерело: {stats['computed_on']}, {stats['n_windows']} вікон, "
        f"учасники {stats['subjects']} */",
        "#define HAR_NORM_MEAN { "
        + ", ".join(c_float(v) for v in stats["mean"]) + " }",
        "#define HAR_NORM_STD  { "
        + ", ".join(c_float(v) for v in stats["std"]) + " }",
        "",
        f"#endif /* {guard} */",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--out", type=Path, help="куди писати (типово firmware.header_out)")
    ap.add_argument("--print", dest="to_stdout", action="store_true",
                    help="вивести в stdout і нічого не писати")
    args = ap.parse_args()

    cfg = load_config(args.config)
    stats_path = ROOT / cfg["paths"]["processed"] / "norm_stats.json"
    if not stats_path.exists():
        print(f"Немає {stats_path}. Спочатку: python scripts/run_all.py",
              file=sys.stderr)
        return 1
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    if stats["channels"] != cfg["channels"]:
        print(f"! канали в norm_stats.json {stats['channels']} не збігаються з "
              f"config.yaml {cfg['channels']}", file=sys.stderr)
        return 1

    name = cfg["firmware"]["header_name"]
    guard = name.upper().replace(".", "_").replace("-", "_")
    text = build_header(cfg, stats, guard)

    if args.to_stdout:
        print(text)
        return 0

    out = args.out or (ROOT / cfg["firmware"]["core_dir"] / "Inc" / name)
    out = out if out.is_absolute() else (ROOT / out)
    out = out.resolve()
    if not out.parent.exists():
        print(f"Немає теки {out.parent} — вкажи --out або виправ "
              "firmware.core_dir у config.yaml", file=sys.stderr)
        return 1

    old = out.read_text(encoding="utf-8") if out.exists() else None
    out.write_text(text, encoding="utf-8")
    print(f"{'оновлено' if old is not None else 'створено'}: {out}")
    if old is not None and old != text:
        print("  (вміст змінився — перезбери прошивку)")
    fs = cfg["processing"]["target_hz"]
    print(f"  {fs} Гц, вікно {cfg['windows']['size_samples']} x "
          f"{len(cfg['channels'])}, ФНЧ {cfg['firmware']['lowpass_hz']} Гц "
          f"({cfg['firmware']['lowpass_order']}-й порядок)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

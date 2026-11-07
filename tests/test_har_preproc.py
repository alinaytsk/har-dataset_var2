#!/usr/bin/env python3
"""Перевірка прошивкового препроцесингу (har_preproc.c) на ПК.

Збирає модуль прошивки звичайним gcc разом з обгорткою har_preproc_host.c
і звіряє його поведінку з еталоном на scipy:

  1. ФНЧ посемпльно збігається з scipy.signal.sosfilt на тих самих коефіцієнтах;
  2. складова вище зрізу справді давиться (а нижче — проходить);
  3. вікна виходять з правильним кроком і містять саме відфільтровані відліки;
  4. har_preproc_normalize робить рівно (x - mean) / std зі statів train.

Потрібен лише gcc у PATH (ARM-тулчейн не потрібен — код HAL-незалежний).

Запуск:
    python tests/test_har_preproc.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def fail(msg: str) -> None:
    print(f"FAIL {msg}")
    raise SystemExit(1)


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    core = (ROOT / cfg["firmware"]["core_dir"]).resolve()
    src = core / "Src" / "har_preproc.c"
    inc = core / "Inc"
    if not src.exists():
        fail(f"немає {src} — перевір firmware.core_dir у config.yaml")

    gcc = shutil.which("gcc") or shutil.which("cc")
    if gcc is None:
        print("SKIP gcc не знайдено в PATH — перевірку пропущено")
        return 0

    fs = float(cfg["processing"]["target_hz"])
    size = int(cfg["windows"]["size_samples"])
    hop = int(cfg["windows"]["stride_samples"])
    n_ch = len(cfg["channels"])
    cutoff = float(cfg["firmware"]["lowpass_hz"])
    order = int(cfg["firmware"]["lowpass_order"])
    sos = butter(order, cutoff / (fs / 2), btype="low", output="sos")

    stats = json.loads((ROOT / cfg["paths"]["processed"] / "norm_stats.json")
                       .read_text(encoding="utf-8"))
    mean = np.array(stats["mean"], dtype="float64")
    std = np.array(stats["std"], dtype="float64")

    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "har_preproc_host.exe"
        build = subprocess.run(
            [gcc, "-O2", "-std=c11", "-Wall", "-Wextra", f"-I{inc}",
             str(HERE / "har_preproc_host.c"), str(src), "-o", str(exe), "-lm"],
            capture_output=True, text=True)
        if build.returncode != 0:
            fail(f"не збирається:\n{build.stderr}")
        if build.stderr.strip():
            print("попередження компілятора:\n" + build.stderr.strip())
        print("OK   har_preproc.c збирається без помилок")

        # Тестовий сигнал: повільний рух (2 Гц), швидкий (8 Гц) і завада вище
        # зрізу (40 Гц) — саме та смуга, якої немає в навчальних даних.
        rng = np.random.default_rng(0)
        n = size + 3 * hop + 7
        t = np.arange(n) / fs
        x = np.zeros((n, n_ch))
        for c in range(n_ch):
            scale = 1.0 if c < 3 else 50.0
            x[:, c] = scale * (np.sin(2 * np.pi * 2.0 * t + c)
                               + 0.5 * np.sin(2 * np.pi * 8.0 * t)
                               + 0.4 * np.sin(2 * np.pi * 40.0 * t)
                               + 0.02 * rng.standard_normal(n))
        stdin = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in x)

        def run(mode: str) -> list[list[str]]:
            proc = subprocess.run([str(exe), mode], input=stdin,
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                fail(f"режим {mode} впав: {proc.stderr}")
            return [line.split() for line in proc.stdout.splitlines() if line]

        # 1. Фільтр посемпльно.
        rows = run("filter")
        got = np.array([[float(v) for v in r[1:]] for r in rows if r[0] == "F"])
        if got.shape != x.shape:
            fail(f"фільтр повернув {got.shape}, очікувано {x.shape}")
        ref = sosfilt(sos, x, axis=0)
        scale = np.abs(ref).max(axis=0)
        rel = np.abs(got - ref).max(axis=0) / scale
        if rel.max() > 1e-5:
            fail(f"розбіжність зі scipy: {rel.max():.2e} (канал {rel.argmax()})")
        print(f"OK   ФНЧ збігається зі scipy.sosfilt (макс. відносна похибка "
              f"{rel.max():.1e}, float32)")

        # 2. Чи справді давиться смуга вище зрізу.
        def amp(sig: np.ndarray, freq: float) -> float:
            seg = sig[size:, 0]                      # після перехідного процесу
            tt = np.arange(seg.size) / fs
            return 2 * abs(np.sum(seg * np.exp(-2j * np.pi * freq * tt)) / seg.size)

        a2_in, a2_out = amp(x, 2.0), amp(got, 2.0)
        a40_in, a40_out = amp(x, 40.0), amp(got, 40.0)
        keep = a2_out / a2_in
        kill = a40_out / a40_in
        if not 0.9 < keep < 1.1:
            fail(f"корисна смуга 2 Гц послаблена до {keep:.3f}")
        if kill > 0.2:
            fail(f"40 Гц послаблено лише до {kill:.3f} — фільтр не працює")
        print(f"OK   2 Гц проходить ({keep:.3f}), 40 Гц давиться "
              f"({kill:.3f}, {20 * np.log10(kill):.1f} дБ)")

        # 3. Вікна.
        rows = run("window")
        wins = [np.array([float(v) for v in r[2:]]).reshape(size, n_ch)
                for r in rows if r[0] == "W"]
        norms = [np.array([float(v) for v in r[2:]]).reshape(size, n_ch)
                 for r in rows if r[0] == "N"]
        expect_n = (n - size) // hop + 1
        if len(wins) != expect_n:
            fail(f"вікон {len(wins)}, очікувано {expect_n}")
        for k, win in enumerate(wins):
            ref_win = ref[k * hop: k * hop + size]
            err = np.abs(win - ref_win).max() / np.abs(ref_win).max()
            if err > 1e-5:
                fail(f"вікно {k} не збігається з відфільтрованим сигналом "
                     f"(похибка {err:.2e})")
        print(f"OK   вікон {len(wins)}, крок {hop}, вміст збігається з "
              "відфільтрованим сигналом")

        # 4. Нормалізація.
        for k, (win, got_norm) in enumerate(zip(wins, norms)):
            ref_norm = (win - mean) / std
            err = np.abs(got_norm - ref_norm).max()
            if err > 1e-4:
                fail(f"нормалізація вікна {k}: похибка {err:.2e}")
        print("OK   нормалізація = (x - mean) / std зі статистик train")

        # 5. Поведінка на втрачених відліках (паритет з препроцесингом датасету).
        max_gap = int(round(float(cfg["processing"]["max_gap_seconds"]) * fs))
        if max_gap >= hop:
            fail(f"max_gap ({max_gap}) має бути меншим за крок вікна ({hop})")

        def run_gap(m: int) -> list[list[str]]:
            proc = subprocess.run([str(exe), "gap", str(m)], input=stdin,
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                fail(f"режим gap {m} впав: {proc.stderr}")
            return [line.split() for line in proc.stdout.splitlines() if line]

        # Малий провал: кожен відлік «коштує» (m + 1) відліків на сітці,
        # тому вікна виходять раніше, ніж без провалів.
        small = max_gap // 2 or 1
        at = [int(r[2]) for r in run_gap(small) if r[0] == "C"]
        if not at:
            fail(f"з провалом {small} не зібралося жодного вікна")
        expect_first = -(-size // (small + 1))        # стеля size/(small+1)
        if abs(at[0] - expect_first) > 1:
            fail(f"перше вікно на відліку {at[0]}, очікувано ≈{expect_first}")
        print(f"OK   провал {small} відл. добивається інтерполяцією "
              f"(вікно на {at[0]}-му відліку замість {size}-го)")

        # Великий провал: вікно має щоразу починатися з нуля, тож при провалі
        # на кожному відліку повне вікно не збереться ніколи.
        big = max_gap + 1
        rows = run_gap(big)
        if any(r[0] == "W" for r in rows):
            fail(f"провал {big} відл. (більше за поріг {max_gap}) не розірвав вікно — "
                 "вікно зібралося через розрив")
        print(f"OK   провал більший за {max_gap} відл. рве вікно, а не склеює його")

    print("\nУсі перевірки прошивкового препроцесингу пройдено.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

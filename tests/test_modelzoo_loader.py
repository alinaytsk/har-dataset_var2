#!/usr/bin/env python3
"""Перевірка завантажувача для ST model zoo на справжніх файлах датасету.

TensorFlow не потрібен: перевіряється ядро (prepare_realworld_npz_arrays),
яке читає .npz, звіряє класи й канали, контролює спліт за людьми і застосовує
статистики з norm_stats.json. Саме тут ховаються помилки інтеграції, а не в
тонкій tf.data-обгортці над ним.

Запуск:
    python tests/test_modelzoo_loader.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "integrations" / "modelzoo" / "human_activity_recognition" / "tf" / "src" / "datasets"
sys.path.insert(0, str(PKG))

from realworld_npz import CHANNELS, prepare_realworld_npz_arrays  # noqa: E402


def fail(msg: str) -> None:
    print(f"FAIL {msg}")
    raise SystemExit(1)


def expect_error(fragment: str, **kwargs) -> None:
    """Перевіряє, що завантажувач падає з очікуваною помилкою, а не мовчить."""
    try:
        prepare_realworld_npz_arrays(**kwargs)
    except ValueError as exc:
        if fragment.lower() in str(exc).lower():
            return
        fail(f"очікувалася помилка про «{fragment}», отримано: {exc}")
    except Exception as exc:                                  # noqa: BLE001
        fail(f"очікувалася ValueError про «{fragment}», отримано {type(exc).__name__}")
    fail(f"помилки про «{fragment}» не сталося, хоча мала")


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    processed = ROOT / cfg["paths"]["processed"]
    if not (processed / "train.npz").exists():
        print(f"SKIP немає {processed}/train.npz — спочатку python scripts/run_all.py")
        return 0

    classes = [c["name"] for c in sorted(cfg["labels"]["classes"], key=lambda c: c["id"])]
    size = int(cfg["windows"]["size_samples"])
    stats = json.loads((processed / "norm_stats.json").read_text(encoding="utf-8"))

    # 1. Нормальний шлях.
    data = prepare_realworld_npz_arrays(str(processed), classes, size, normalization=True)
    shapes = {k: data[f"x_{k}"].shape for k in ("train", "valid", "test")}
    for key, shape in shapes.items():
        if shape[1:] != (size, len(CHANNELS)):
            fail(f"{key}: форма {shape}, очікувано (N, {size}, {len(CHANNELS)})")
        if data[f"x_{key}"].dtype != np.float32:
            fail(f"{key}: тип {data[f'x_{key}'].dtype}, очікувано float32")
        if not np.isfinite(data[f"x_{key}"]).all():
            fail(f"{key}: є NaN або inf")
    print(f"OK   форми і типи: train{shapes['train']} valid{shapes['valid']} "
          f"test{shapes['test']}")

    # 2. Класи в тому самому порядку, що в конфігу датасету.
    if data["class_names"] != classes:
        fail(f"порядок класів {data['class_names']} != {classes}")
    for key in ("train", "valid", "test"):
        y = data[f"y_{key}"]
        if y.min() < 0 or y.max() >= len(classes):
            fail(f"{key}: мітки поза діапазоном 0..{len(classes) - 1}")
    print(f"OK   класи {classes} у тому самому порядку, що в датасеті")

    # 3. Спліт за людьми збережений.
    subjects = data["subjects"]
    if subjects["train"] != sorted(cfg["split"]["train"]):
        fail(f"train: учасники {subjects['train']} != {sorted(cfg['split']['train'])}")
    if subjects["val"] != sorted(cfg["split"]["val"]):
        fail(f"val: учасники {subjects['val']}")
    if subjects["test"] != sorted(cfg["split"]["test"]):
        fail(f"test: учасники {subjects['test']}")
    print(f"OK   спліт за людьми: train={subjects['train']} val={subjects['val']} "
          f"test={subjects['test']}")

    # 4. Нормалізація застосована саме збереженими статистиками.
    flat = data["x_train"].reshape(-1, len(CHANNELS))
    if abs(float(flat.mean())) > 0.02 or abs(float(flat.std()) - 1.0) > 0.05:
        fail(f"train після нормалізації: mean={flat.mean():.4f}, std={flat.std():.4f}")
    raw = prepare_realworld_npz_arrays(str(processed), classes, size, normalization=False)
    mean = np.asarray(stats["mean"], dtype="float32")
    std = np.asarray(stats["std"], dtype="float32")
    expected = (raw["x_test"][:64] - mean) / std
    if not np.allclose(expected, data["x_test"][:64], atol=1e-5):
        fail("нормалізація не збігається з (x - mean) / std із norm_stats.json")
    print("OK   нормалізація = статистики train з norm_stats.json")

    # 5. Без нормалізації дані лишаються фізичними (перевірка одиниць).
    lying = classes.index("Lying")
    mag = float(np.linalg.norm(raw["x_test"][raw["y_test"] == lying][:, :, :3],
                               axis=2).mean())
    if not 0.85 <= mag <= 1.15:
        fail(f"|acc| для Lying = {mag:.3f} g — одиниці поїхали")
    print(f"OK   без нормалізації дані у фізичних одиницях (|acc| Lying = {mag:.3f} g)")

    # 6. Помилки конфігу мають бути гучними, а не тихими.
    expect_error("class_names", training_path=str(processed),
                 class_names=["Walking", "Lying", "Stationary", "Running", "Stairs"],
                 sequence_length=size, normalization=False)
    expect_error("input_shape", training_path=str(processed), class_names=classes,
                 sequence_length=size // 2, normalization=False)
    expect_error("немає теки", training_path=str(ROOT / "no_such_dir"),
                 class_names=classes, sequence_length=size, normalization=False)
    print("OK   неправильний порядок класів, довжина вікна і шлях дають зрозумілу помилку")

    print("\nЗавантажувач для model zoo готовий до передачі.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

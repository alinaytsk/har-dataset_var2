"""Завантажувач готового датасету har-dataset (.npz) для ST model zoo.

Навіщо окремий завантажувач. `realworld_imu.py` читає один склеєний CSV і сам
ріже його на вікна, сам ділить на вибірки і сам рахує нормалізацію. Наш датасет
усе це вже зробив — і зробив суворіше:

  * вікна нарізані так, що жодне не перетинає розрив у даних, межу запису,
    активності чи учасника;
  * перекриття 50 % (у CSV-завантажувачі вікна не перекриваються);
  * спліт за людьми ФІКСОВАНИЙ і записаний у config.yaml датасету, а не
    випадковий — щоб результати різних запусків можна було порівнювати;
  * mean/std пораховані тільки на train і лежать у norm_stats.json; ті самі
    числа зашиваються в прошивку, тому перераховувати їх не можна.

Тому цей модуль нічого не перераховує: він читає готові train/val/test.npz,
застосовує збережені статистики і віддає tf.data.Dataset у тому самому
вигляді (batch, time, axes, 1), що й решта завантажувачів цього репозиторію.

TensorFlow імпортується всередині функції, щоб ядро (prepare_*) можна було
перевіряти без нього.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
SPLITS = ("train", "val", "test")


def _resolve(training_path: str) -> Path:
    """Приймає і теку data/processed, і шлях до train.npz."""
    path = Path(training_path).expanduser()
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        raise ValueError(f"RealWorld npz: немає теки {path}. Вкажи шлях до "
                         "data/processed з репозиторію har-dataset.")
    missing = [f"{name}.npz" for name in SPLITS if not (path / f"{name}.npz").exists()]
    if missing:
        raise ValueError(f"RealWorld npz: у {path} бракує файлів {missing}. "
                         "Згенеруй їх командою python scripts/run_all.py.")
    return path


def prepare_realworld_npz_arrays(training_path: str, class_names: List[str],
                                 sequence_length: int, normalization: bool = True):
    """Читає готові вибірки. Нічого не ріже, не ділить і не перераховує.

    Повертає словник із x_/y_ по трьох вибірках, mean/std і назвами класів.
    """
    root = _resolve(training_path)
    data, subjects = {}, {}
    file_classes = None

    for name in SPLITS:
        with np.load(root / f"{name}.npz", allow_pickle=False) as npz:
            x = npz["X"].astype("float32")
            y = npz["y"].astype("int64")
            subjects[name] = sorted({int(s) for s in npz["subject"]})
            names = [str(c) for c in npz["class_names"]]
            channels = [str(c) for c in npz["channels"]]

        if file_classes is None:
            file_classes = names
        elif names != file_classes:
            raise ValueError(f"У {name}.npz інший порядок класів: {names}")
        if channels != CHANNELS:
            raise ValueError(f"Очікувані канали {CHANNELS}, у файлі {channels}")
        if x.shape[1] != sequence_length:
            raise ValueError(
                f"input_shape[0]={sequence_length}, а у {name}.npz вікно "
                f"{x.shape[1]} відліків. Або виправ input_shape у конфігу, або "
                "перегенеруй датасет з іншим windows.size_samples.")
        data[name] = (x, y)

    # Порядок класів у конфігу мусить збігатися з порядком у файлах: інакше
    # точність буде правильна, а підписи класів на платі — зсунуті.
    if list(class_names) != file_classes:
        raise ValueError(
            "class_names у конфігу не збігаються з датасетом.\n"
            f"  конфіг: {list(class_names)}\n"
            f"  датасет: {file_classes}")

    # Контроль головної гарантії датасету — спліт за людьми.
    for a in SPLITS:
        for b in SPLITS:
            if a < b and set(subjects[a]) & set(subjects[b]):
                raise ValueError(f"учасники перетинаються між {a} і {b}: "
                                 f"{sorted(set(subjects[a]) & set(subjects[b]))}")

    mean = np.zeros(len(CHANNELS), dtype="float32")
    std = np.ones(len(CHANNELS), dtype="float32")
    if normalization:
        stats_file = root / "norm_stats.json"
        if not stats_file.exists():
            raise ValueError(f"немає {stats_file} — без нього нормалізувати нічим")
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
        if stats["channels"] != CHANNELS:
            raise ValueError(f"norm_stats.json описує інші канали: {stats['channels']}")
        mean = np.asarray(stats["mean"], dtype="float32")
        std = np.asarray(stats["std"], dtype="float32")
        for name in SPLITS:
            x, y = data[name]
            data[name] = ((x - mean) / std, y)

    out = {"mean": mean, "std": std, "class_names": list(file_classes),
           "subjects": subjects}
    for name, key in zip(SPLITS, ("train", "valid", "test")):
        out[f"x_{key}"], out[f"y_{key}"] = data[name]
    return out


def load_realworld_npz(training_path: str, class_names: List[str],
                       input_shape: Tuple[int], batch_size: int, seed: int,
                       normalization: bool = True, to_cache: bool = False,
                       **kwargs):
    """Три tf.data.Dataset у розкладці (batch, time, axes, 1).

    validation_split і test_split ігноруються свідомо: спліт зафіксований у
    самих файлах (за людьми), і підмінювати його випадковим не можна.
    """
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical

    if not training_path:
        return None, None, None
    shape = tuple(int(v) for v in input_shape)
    if len(shape) != 3 or shape[1:] != (len(CHANNELS), 1):
        raise ValueError(f"RealWorld npz очікує (timesteps, {len(CHANNELS)}, 1), "
                         f"отримано {input_shape}")

    data = prepare_realworld_npz_arrays(training_path, class_names, shape[0],
                                        normalization)

    def make_ds(split: str, shuffle: bool):
        x = data[f"x_{split}"][..., np.newaxis]
        y = to_categorical(data[f"y_{split}"], num_classes=len(class_names))
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        return ds.cache() if to_cache else ds

    print(f"[realworld_npz] train={len(data['x_train'])} "
          f"valid={len(data['x_valid'])} test={len(data['x_test'])} вікон, "
          f"вікно {shape[0]} x {len(CHANNELS)}, "
          f"нормалізація={'norm_stats.json' if normalization else 'немає'}")
    print(f"[realworld_npz] учасники: {data['subjects']}")

    return make_ds("train", True), make_ds("valid", True), make_ds("test", False)

#!/usr/bin/env python3
"""Крок 1/3: завантаження RealWorld (HAR) 2016 з офіційного сайту.

Сайт роздає окремі csv-архіви на кожну пару (сенсор, активність) для кожного
учасника:  {base_url}/s{N}/data/{sensor}_{activity}_csv.zip
Тому качаємо лише acc + gyr для потрібних активностей — це ~0.7 ГБ замість
3.5 ГБ повного датасету (gps, light, magnetic, sound не потрібні).

Усередині кожного архіву лежать усі 7 позицій на тілі; розпаковуємо тільки
dataset.position (upperarm), решту не витягуємо. `--all-positions` знімає фільтр.

Усі параметри — з config.yaml. Приклади:
    python scripts/download.py
    python scripts/download.py --subjects 1,2
    python scripts/download.py --check-only      # лише перевірити доступність
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
UA = "har-dataset/1.0 (student project; python-urllib)"
CHUNK = 1 << 20


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_list(spec: str | None, default: list):
    """'1-3,7' -> [1,2,3,7]; 'walking,lying' -> ['walking','lying']."""
    if not spec:
        return list(default)
    out: list = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if part.isdigit():
            out.append(int(part))
        elif "-" in part and all(p.isdigit() for p in part.split("-", 1)):
            lo, hi = (int(p) for p in part.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(part)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def head(url: str, timeout: float = 30.0) -> int | None:
    """Розмір файла за HEAD-запитом, або None якщо недоступний."""
    req = Request(url, headers={"User-Agent": UA}, method="HEAD")
    try:
        with urlopen(req, timeout=timeout) as resp:
            size = resp.headers.get("Content-Length")
            return int(size) if size else -1
    except (HTTPError, URLError):
        return None


def fetch(url: str, dest: Path, timeout: float = 120.0, retries: int = 3) -> bool:
    """Качає url у dest із докачуванням і повторами."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": UA}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as resp:
                if offset and getattr(resp, "status", 200) != 206:
                    offset = 0                       # сервер не підтримав докачування
                with part.open("ab" if offset else "wb") as fh:
                    shutil.copyfileobj(resp, fh, CHUNK)
            part.replace(dest)
            return True
        except HTTPError as exc:
            if exc.code == 404:
                return False
            print(f"    HTTP {exc.code}, спроба {attempt}/{retries}", file=sys.stderr)
        except (URLError, TimeoutError, OSError) as exc:
            print(f"    {exc}, спроба {attempt}/{retries}", file=sys.stderr)
        time.sleep(2 * attempt)
    return False


def extract(archive: Path | io.BytesIO, target: Path, position: str | None,
            depth: int = 0) -> list[str]:
    """Розпаковує csv (лише потрібну позицію, якщо задана).

    Частина архівів (сходи в s4, s7, s14) містить ВКЛАДЕНІ zip — по одному на
    частину запису (`_1`, `_2`, `_3`), тому заходимо в них рекурсивно.
    Захист від zip-slip: архів з інтернету — недовірені дані.
    """
    if depth > 3:
        return []
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    written: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            lower = info.filename.lower()
            if lower.endswith(".zip"):               # вкладений архів -> у пам'ять
                with zf.open(info) as src:
                    nested = io.BytesIO(src.read())
                written += extract(nested, target, position, depth + 1)
                continue
            if not lower.endswith(".csv"):
                continue
            if position and position not in lower:
                continue
            out = target / Path(info.filename).name
            if root != out.resolve().parent:
                print(f"    ! підозрілий шлях, пропущено: {info.filename}",
                      file=sys.stderr)
                continue
            with zf.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            written.append(out.name)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--subjects", help="наприклад 1-3,7 (типово — усі з config.yaml)")
    ap.add_argument("--activities", help="наприклад walking,lying")
    ap.add_argument("--all-positions", action="store_true",
                    help="розпакувати всі позиції на тілі, не лише upperarm")
    ap.add_argument("--delete-archives", action="store_true",
                    help="видаляти zip після розпаковки (економить ~0.7 ГБ)")
    ap.add_argument("--check-only", action="store_true",
                    help="лише перевірити доступність і сумарний розмір")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    raw = ROOT / cfg["paths"]["raw"]
    subjects = parse_list(args.subjects, ds["subjects"])
    activities = parse_list(args.activities, ds["activities_used"])
    sensors = {k: v["archive"] for k, v in ds["sensors"].items()}
    position = None if args.all_positions else ds["position"].lower()

    print(f"RealWorld (HAR) 2016 -> {raw}")
    print(f"учасники: {subjects}")
    print(f"активності: {activities}   сенсори: {list(sensors)}")
    print(f"позиція: {position or 'усі'}\n")

    if args.check_only:
        total, missing = 0, []
        for s in subjects:
            for act in activities:
                for prefix in sensors.values():
                    url = f"{ds['base_url']}/s{s}/data/{prefix}_{act}_csv.zip"
                    size = head(url)
                    if size is None:
                        missing.append(f"s{s}/{prefix}_{act}")
                    elif size > 0:
                        total += size
            print(f"  s{s}: перевірено")
        print(f"\nсумарно до завантаження: {human(total)}")
        print(f"недоступних архівів: {len(missing)}")
        for m in missing:
            print(f"  - {m}")
        return 1 if missing else 0

    manifest_path = raw / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    failed: list[str] = []
    downloaded = skipped = 0
    for s in subjects:
        print(f"[s{s}]")
        for act in activities:
            for sensor, prefix in sensors.items():
                key = f"s{s}/{prefix}_{act}_csv.zip"
                url = f"{ds['base_url']}/s{s}/data/{prefix}_{act}_csv.zip"
                archive = raw / "_archives" / f"s{s}" / f"{prefix}_{act}_csv.zip"
                target = raw / f"s{s}"

                have_csv = manifest.get(key, {}).get("extracted")
                if archive.exists() or (have_csv and args.delete_archives):
                    if have_csv and all((target / n).exists() for n in have_csv):
                        skipped += 1
                        continue
                if not archive.exists() and not fetch(url, archive):
                    print(f"  ! не завантажено: {key}", file=sys.stderr)
                    failed.append(key)
                    continue

                try:
                    names = extract(archive, target, position)
                except zipfile.BadZipFile:
                    print(f"  ! битий архів, видаляю і повторю наступного разу: {key}",
                          file=sys.stderr)
                    archive.unlink(missing_ok=True)
                    failed.append(key)
                    continue

                manifest[key] = {"url": url, "sha256": sha256(archive),
                                 "bytes": archive.stat().st_size, "extracted": names}
                downloaded += 1
                print(f"  {prefix}_{act}: {human(archive.stat().st_size)} -> {names}")
                if args.delete_archives:
                    archive.unlink()
                    manifest[key]["archive_kept"] = False

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                 encoding="utf-8")

    csv_files = sorted(raw.glob("s*/*.csv"))
    size = sum(p.stat().st_size for p in csv_files)
    print(f"\nзавантажено архівів: {downloaded}, пропущено (вже є): {skipped}")
    print(f"csv у {raw}: {len(csv_files)} файлів, {human(size)}")
    print(f"manifest: {manifest_path}")

    if failed:
        print(f"\nНе вдалося завантажити {len(failed)} архівів:", file=sys.stderr)
        for key in failed:
            print(f"  - {key}", file=sys.stderr)
        print("Перезапусти скрипт — уже завантажене пропускається.", file=sys.stderr)
        return 1
    print("Готово. Далі: python scripts/preprocess.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

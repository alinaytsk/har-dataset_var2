#!/usr/bin/env python3
"""Одна команда для всього пайплайна: завантаження -> обробка -> вікна -> перевірки.

    python scripts/run_all.py                  # повний прогін
    python scripts/run_all.py --subjects 1,2   # швидкий прогін на двох учасниках
    python scripts/run_all.py --skip-download  # якщо data/raw уже заповнена

Кожен крок — окремий процес; якщо крок падає, пайплайн зупиняється з його кодом.
Seed і всі параметри беруться з config.yaml, тому прогін відтворюваний.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(script: str, extra: list[str]) -> int:
    cmd = [sys.executable, str(SCRIPTS / script), *extra]
    print(f"\n{'═' * 78}\n▶ {' '.join(cmd[1:])}\n{'═' * 78}", flush=True)
    started = time.time()
    code = subprocess.call(cmd, cwd=ROOT)
    print(f"\n← {script}: код {code}, {time.time() - started:.1f} с", flush=True)
    return code


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", help="обмежити список учасників, наприклад 1,2")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--verify-rerun", action="store_true",
                    help="у перевірках ще раз перезапустити нарізку і порівняти хеші")
    args = ap.parse_args()

    subj = ["--subjects", args.subjects] if args.subjects else []
    steps: list[tuple[str, list[str]]] = []
    if not args.skip_download:
        steps.append(("download.py", subj))
    steps += [("preprocess.py", subj), ("make_windows.py", []),
              ("checks.py", ["--verify-rerun"] if args.verify_rerun else [])]

    started = time.time()
    for script, extra in steps:
        code = run(script, extra)
        if code != 0:
            print(f"\nПайплайн зупинено на {script} (код {code}).", file=sys.stderr)
            return code

    print(f"\n{'═' * 78}")
    print(f"Готово за {(time.time() - started) / 60:.1f} хв.")
    print("Дані: data/processed/  ·  звіти: reports/checks.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""LOCATA recording을 AGG-RL 평가용 sample_input 형식 .pkl로 내보낸다.

LOCATA 배열을 논문의 평가셋에 매핑한다:
    benchmark2 (12ch) -> NAO robot
    eigenmike  (32ch) -> Eigenmike

논문 4.2절에 맞춰, 정적 음원이 max_speakers 이하인 recording만 16 kHz로 리샘플해
내보낸다.

사용법:
    python export_locata.py \
        --locata_root ./datasets/locata/dev \
        --array benchmark2 \
        --out_dir ./locata_eval/nao \
        --max_speakers 2

이후 inference.py의 glob를 출력 디렉터리로 지정하면 된다.
"""
from __future__ import annotations

import argparse
import os
import pickle as pkl

import glob as _glob

import numpy as np

from data.locata import (
    LOCATA_ARRAYS,
    _cols,
    _read_position_table,
    find_recordings,
    load_recording,
    to_dataframe,
)


def _is_static(rec_dir: str, move_thresh_m: float = 0.1) -> bool:
    """recording의 모든 음원이 정적(이동 < move_thresh_m)이면 True.

    논문 4.2절은 non-moving source만 평가에 사용한다. LOCATA task3~task6은
    음원이 수 m 이동하므로 제외되고, task1/task2의 정적 recording만 남는다.
    """
    for sp in sorted(_glob.glob(os.path.join(rec_dir, "position_source_*.txt"))):
        xyz = _cols(_read_position_table(sp), "x", "y", "z")  # (3, N)
        if float(np.linalg.norm(xyz.max(1) - xyz.min(1))) >= move_thresh_m:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locata_root", required=True,
                    help="LOCATA dev/ 또는 eval/ 트리 루트 경로")
    ap.add_argument("--array", required=True, choices=sorted(LOCATA_ARRAYS),
                    help="LOCATA 배열 이름 (benchmark2=NAO robot, eigenmike=Eigenmike)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_speakers", type=int, default=2)
    ap.add_argument("--static_only", action="store_true", default=True,
                    help="이동 음원 recording을 건너뛴다 (논문 4.2절 기본 동작)")
    ap.add_argument("--include_moving", dest="static_only", action="store_false",
                    help="이동 음원 recording도 포함")
    args = ap.parse_args()

    paper_name, n_ch = LOCATA_ARRAYS[args.array]
    print(f"array={args.array} ({paper_name}, {n_ch}ch)  root={args.locata_root}")
    os.makedirs(args.out_dir, exist_ok=True)

    n_ok = n_skip = n_moving = 0
    for i, rec_dir in enumerate(find_recordings(args.locata_root, args.array)):
        if args.static_only and not _is_static(rec_dir):
            n_moving += 1
            continue
        sample = load_recording(rec_dir, args.array, max_speakers=args.max_speakers)
        if sample is None:
            n_skip += 1
            continue
        df = to_dataframe(sample)
        out_path = os.path.join(args.out_dir, f"{i}.pkl")
        with open(out_path, "wb") as f:
            pkl.dump(df, f)
        n_ok += 1
        print(f"  wrote {out_path}  n_spk={sample['n_spk']}  "
              f"T={sample['input_audio'].shape[1]}  from {os.path.relpath(rec_dir, args.locata_root)}")

    print(f"\nexported {n_ok} recordings, skipped {n_skip} "
          f"(>{args.max_speakers} speakers or missing files), "
          f"skipped {n_moving} moving-source recordings")


if __name__ == "__main__":
    main()

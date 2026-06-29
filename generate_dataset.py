"""합성 샘플을 AGG-RL용 ``sample_input`` 형식의 .pkl 파일로 덤프한다.

GI-DOAEnet 데이터 파이프라인(``data/simulate.py`` + ``data/mic_arrays.py``)을
재사용하며, ``sample_input/<C>/<idx>.pkl``과 완전히 동일한 스키마의 한 행짜리
pandas DataFrame을 저장한다. 따라서 ``inference.py``로 바로 로드할 수 있다:

    vad                (n_spk, T)        bool
    n_spk              int
    n_channel          int
    mic_dim            str   ('D3')
    input_audio        (C, T)            float32
    mic_coordinate     (C, 3)            float64
    spherical_position (n_spk, 3, T)     float64

사용법:
    python generate_dataset.py \
        --librispeech_root /path/to/LibriSpeech/test-clean \
        --ms_snsd_root     /path/to/MS-SNSD/noise_test \
        --out_dir ./generated \
        --channels 4 8 12 \
        --per_channel 10 \
        --profile dynamic4to12

의존성: gpuRIR, soundfile, webrtcvad, pandas (GI-DOAEnet requirements 참고).
"""
from __future__ import annotations

import argparse
import os
import pickle as pkl

import numpy as np
import pandas as pd

from data.dataset import (
    SyntheticDOADataset,
    _load_audio_mono,
    _resample_audio,
    _crop_or_pad,
    _sample_vad_mask,
    _discover_audio_files,
)
from data.simulate import SimulationConfig, simulate_one_sample, N_SPK
from data.mic_arrays import get_fixed_array, sample_dynamic_array, random_rotate


def _to_sample_input_row(sample: dict, vad_full: np.ndarray) -> pd.DataFrame:
    n_spk = int(sample["n_spk"])
    n_channel = int(sample["input_audio"].shape[0])

    # sample_input과 정확히 맞추기 위해 패딩(비활성) 화자를 잘라낸다
    vad = vad_full[:n_spk].astype(bool)                              # (n_spk, T)
    spherical = sample["spherical_position"][:n_spk].astype(np.float64)  # (n_spk,3,T)

    row = {
        "vad": [vad],
        "n_spk": [n_spk],
        "n_channel": [n_channel],
        "mic_dim": ["D3"],
        "input_audio": [sample["input_audio"].astype(np.float32)],
        "mic_coordinate": [sample["mic_coordinate"].astype(np.float64)],
        "spherical_position": [spherical],
    }
    return pd.DataFrame(row)


def _sample_mic_coords(profile_array: str, C: int, rng, rotate: bool) -> np.ndarray:
    if profile_array == "dynamic":
        coords = sample_dynamic_array(C, rng=rng)
    else:
        coords = get_fixed_array(profile_array)
    if rotate:
        coords = random_rotate(coords, rng)
    return coords.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech_root", required=True)
    ap.add_argument("--ms_snsd_root", required=True)
    ap.add_argument("--out_dir", default="./generated")
    ap.add_argument("--channels", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--per_channel", type=int, default=10)
    ap.add_argument("--array", default="dynamic",
                    choices=["dynamic", "tetrahedron", "nao4", "nao12"])
    ap.add_argument("--no_rotate", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = SimulationConfig()
    speech_files = _discover_audio_files(args.librispeech_root, ("*.flac", "*.wav"))
    noise_files = _discover_audio_files(args.ms_snsd_root, ("*.wav", "*.flac"))
    if not speech_files:
        raise FileNotFoundError(f"No speech under {args.librispeech_root!r}")
    if not noise_files:
        raise FileNotFoundError(f"No noise under {args.ms_snsd_root!r}")
    print(f"speech files: {len(speech_files)} | noise files: {len(noise_files)}")

    def load_audio(path, rng):
        wav, sr = _load_audio_mono(path)
        wav = _resample_audio(wav, sr, cfg.sample_rate)
        return _crop_or_pad(wav, cfg.segment_samples, rng)

    for C in args.channels:
        ch_dir = os.path.join(args.out_dir, str(C))
        os.makedirs(ch_dir, exist_ok=True)
        for i in range(args.per_channel):
            rng = np.random.default_rng(args.seed * 100_003 + C * 1009 + i)

            sp_idx = rng.choice(len(speech_files), size=N_SPK, replace=False)
            speeches, vad = [], []
            for j in sp_idx:
                s = load_audio(speech_files[j], rng)
                speeches.append(s)
                vad.append(_sample_vad_mask(s, cfg.sample_rate))
            vad = np.stack(vad, axis=0).astype(np.float32)

            noise = load_audio(noise_files[int(rng.integers(0, len(noise_files)))], rng)
            mic = _sample_mic_coords(args.array, C, rng, not args.no_rotate)

            sample = simulate_one_sample(
                speeches=speeches, coherent_noise=noise, mic_coords=mic,
                rng=rng, config=cfg,
            )
            vad[int(sample["n_spk"]):] = 0.0

            df = _to_sample_input_row(sample, vad)
            out_path = os.path.join(ch_dir, f"{i}.pkl")
            with open(out_path, "wb") as f:
                pkl.dump(df, f)
            print(f"wrote {out_path}  (n_spk={int(sample['n_spk'])})")


if __name__ == "__main__":
    main()

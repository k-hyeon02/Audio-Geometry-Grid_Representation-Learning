"""AGG-RL 평가를 위한 LOCATA 실제 녹음 로더.

논문(Table 2)은 LOCATA 코퍼스(Loellmann et al., 2018)의 실제 마이크 배열 2종으로
평가한다:

    NAO robot  -> LOCATA 배열 "benchmark2" (12 ch)   [seen geometry]
    Eigenmike  -> LOCATA 배열 "eigenmike"  (32 ch)   [unseen geometry]

논문 4.2절에 따라 정적(non-moving) 음원, 최대 2화자, 16 kHz 리샘플 조건만 사용한다.
이 모듈은 LOCATA recording 디렉터리를 읽어 AGG-RL 스키마(``sample_input/*.pkl`` /
``data.simulate.simulate_one_sample``와 동일)로 샘플을 만든다:

    input_audio        (C, T)            float32
    vad                (n_spk, T)        float32/bool
    mic_coordinate     (C, 3)            float32   (미터, 배열 중심 기준)
    spherical_position (n_spk, 3, T)     float64   [azimuth_deg, elevation_deg, dist_m]
    n_spk, n_channel   int
    mic_dim            'D3'

정답 DOA는 LOCATA 컨벤션(sap_locata_eval/get_truth.m)을 따른다:
    v = R^T (h - p)      # 음원 위치 h, 배열 위치 p, 배열 회전 R
그 후 모델 컨벤션(model.util.cart2sph)으로 구면 좌표로 변환해 AGG-RL이 기대하는
라벨 형식에 맞춘다.

LOCATA recording 디렉터리 구조(하나의 task/recording):
    audio_array_<name>.wav        # (T, C) 다채널 배열 녹음
    position_array_<name>.txt     # 배열 위치 + 회전 + 마이크 좌표 (프레임별)
    position_source_<name>.txt    # 음원별 위치 (프레임별)
    audio_source_<name>.wav       # clean close-talk 기준 신호 (VAD 계산용)
    required_time.txt             # 평가 타임스탬프 + valid flag
"""
from __future__ import annotations

import os
from glob import glob

import numpy as np
import pandas as pd


# LOCATA 배열 이름 -> (논문 표기, 채널 수)
LOCATA_ARRAYS = {
    "benchmark2": ("NAO robot", 12),
    "eigenmike": ("Eigenmike", 32),
    "dicit": ("DICIT", 15),
    "dummy": ("dummy", 4),
}

FS_TARGET = 16000


# ---------------------------------------------------------------------------
# 저수준 reader
# ---------------------------------------------------------------------------
def _read_wav(path: str):
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
    return data, sr


def _resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x.astype(np.float32, copy=False)
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, target)
    return resample_poly(x, target // g, sr // g, axis=0).astype(np.float32)


def _read_position_table(path: str) -> pd.DataFrame:
    """LOCATA position_*.txt는 헤더 행이 있는 공백/콤마 구분 테이블이다."""
    # 파일은 탭/공백 + 헤더 형식이므로 pandas가 구분자를 자동 감지하게 둔다
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def _cols(df: pd.DataFrame, *names) -> np.ndarray:
    """지정한 컬럼들을 (3, N) float 배열로 쌓아 반환한다."""
    return np.stack([df[n].to_numpy(dtype=np.float64) for n in names], axis=0)


# ---------------------------------------------------------------------------
# 지오메트리 / 정답(GT)
# ---------------------------------------------------------------------------
def _array_mic_coords(pos_array: pd.DataFrame, n_channels: int) -> np.ndarray:
    """배열 중심 기준의 마이크별 좌표(미터)를 추출한다.

    LOCATA는 절대 마이크 위치를 mic1_x, mic1_y, mic1_z, ... 컬럼으로 저장한다
    (시간에 따라 변할 수도 있음). 정적 녹음이므로 첫 프레임을 취하고 마이크
    무게중심으로 정규화한다 -- 합성 데이터의 ``mic_coordinate``가 배열 중심
    기준인 것과 동일하게 맞춘다.
    """
    mics = []
    for m in range(1, n_channels + 1):
        cx, cy, cz = f"mic{m}_x", f"mic{m}_y", f"mic{m}_z"
        if cx not in pos_array.columns:
            raise KeyError(f"position_array 파일에 마이크 컬럼 {cx}가 없음")
        mics.append([
            pos_array[cx].to_numpy(dtype=np.float64)[0],
            pos_array[cy].to_numpy(dtype=np.float64)[0],
            pos_array[cz].to_numpy(dtype=np.float64)[0],
        ])
    coords = np.asarray(mics, dtype=np.float64)        # (C, 3) 절대좌표
    coords -= coords.mean(axis=0, keepdims=True)       # 배열 중심으로 정규화
    return coords.astype(np.float32)


def _rotation_at(pos_array: pd.DataFrame, t: int) -> np.ndarray:
    """프레임 t에서의 3x3 배열 회전 행렬(rotation 컬럼들로 구성)."""
    # LOCATA는 회전을 rotation* 9개 컬럼(또는 회전 행렬)으로 제공한다.
    rot_cols = [c for c in pos_array.columns if c.startswith("rotation")]
    if len(rot_cols) >= 9:
        vals = [pos_array[c].to_numpy(dtype=np.float64)[t] for c in rot_cols[:9]]
        return np.asarray(vals, dtype=np.float64).reshape(3, 3)
    # fallback: 단위행렬 (이미 배열 로컬 좌표인 경우)
    return np.eye(3, dtype=np.float64)


def compute_gt_doa(array_pos: np.ndarray, array_rot: np.ndarray,
                   source_pos: np.ndarray):
    """배열 기준 음원의 DOA를 모델 컨벤션으로 계산한다.

    array_pos  : (3,)  배열 기준 위치
    array_rot  : (3,3) 배열 회전 행렬 R
    source_pos : (3,)  음원 위치
    model.util.cart2sph 컨벤션으로 (azimuth_deg, elevation_deg, distance_m) 반환.
    """
    v = array_rot.T @ (source_pos - array_pos)          # LOCATA: v = R^T (h - p)
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    az = np.degrees(np.arctan2(y, x))                   # model.util.cart2sph와 동일
    el = np.degrees(np.pi / 2 - np.arctan2(z, np.sqrt(x * x + y * y)))
    dist = float(np.sqrt(x * x + y * y + z * z))
    return float(az % 360.0), float(el), dist


# ---------------------------------------------------------------------------
# clean 기준 신호에서 VAD 추출
# ---------------------------------------------------------------------------
def _vad_from_source(audio_source: np.ndarray, fs: int) -> np.ndarray:
    """clean close-talk 음원에서 샘플 단위 VAD를 계산한다(에너지 기반)."""
    frame = int(fs * 0.03)
    out = np.zeros(audio_source.shape[0], dtype=np.float32)
    peak = np.max(np.abs(audio_source)) + 1e-8
    thr = peak * (10 ** (-40 / 20))
    for s in range(0, len(audio_source) - frame + 1, frame):
        if np.sqrt(np.mean(audio_source[s:s + frame] ** 2)) > thr:
            out[s:s + frame] = 1.0
    return out


# ---------------------------------------------------------------------------
# 메인 로더
# ---------------------------------------------------------------------------
def load_recording(rec_dir: str, array_name: str, max_speakers: int = 2):
    """지정한 배열의 LOCATA recording 하나를 AGG-RL 스키마로 읽어온다.

    sample_input 형식의 dict를 반환하며, 음원이 ``max_speakers``보다 많으면
    None을 반환한다(논문은 최대 2화자까지만 사용).
    """
    if array_name not in LOCATA_ARRAYS:
        raise ValueError(f"알 수 없는 LOCATA 배열 {array_name!r}")
    _, n_channels = LOCATA_ARRAYS[array_name]

    audio_path = os.path.join(rec_dir, f"audio_array_{array_name}.wav")
    parr_path = os.path.join(rec_dir, f"position_array_{array_name}.txt")
    if not (os.path.exists(audio_path) and os.path.exists(parr_path)):
        return None

    audio, sr = _read_wav(audio_path)                   # (T, C)
    audio = _resample(audio, sr, FS_TARGET)
    audio = audio.T.copy()                              # (C, T)
    T = audio.shape[1]

    pos_array = _read_position_table(parr_path)
    mic_coord = _array_mic_coords(pos_array, n_channels)
    array_p = _cols(pos_array, "x", "y", "z")[:, 0]     # (3,) 정적 -> 0번 프레임
    array_R = _rotation_at(pos_array, 0)

    # 음원들
    src_paths = sorted(glob(os.path.join(rec_dir, "position_source_*.txt")))
    n_spk = len(src_paths)
    if n_spk == 0 or n_spk > max_speakers:
        return None

    spherical = np.zeros((max_speakers, 3, T), dtype=np.float64)
    vad = np.zeros((max_speakers, T), dtype=np.float32)
    for i, sp_path in enumerate(src_paths):
        sp = _read_position_table(sp_path)
        src_p = _cols(sp, "x", "y", "z")[:, 0]          # 정적 음원
        az, el, dist = compute_gt_doa(array_p, array_R, src_p)
        spherical[i, 0, :] = az
        spherical[i, 1, :] = el
        spherical[i, 2, :] = dist

        # clean close-talk 기준 신호가 있으면 그것으로 VAD 계산
        src_tag = os.path.basename(sp_path).replace("position_source_", "").replace(".txt", "")
        asrc = os.path.join(rec_dir, f"audio_source_{src_tag}.wav")
        if os.path.exists(asrc):
            a, asr = _read_wav(asrc)
            a = _resample(a[:, 0:1], asr, FS_TARGET)[:, 0]
            v = _vad_from_source(a, FS_TARGET)
            vad[i, :min(T, len(v))] = v[:min(T, len(v))]
        else:
            vad[i, :] = 1.0                              # 기준 신호 없으면 항상 활성으로 가정

    return {
        "vad": vad[:n_spk].astype(bool),
        "n_spk": int(n_spk),
        "n_channel": int(n_channels),
        "mic_dim": "D3",
        "input_audio": audio.astype(np.float32),
        "mic_coordinate": mic_coord.astype(np.float64),
        "spherical_position": spherical[:n_spk].astype(np.float64),
    }


def find_recordings(locata_root: str, array_name: str):
    """LOCATA dev/eval 트리에서 해당 배열의 recording 디렉터리들을 순회 반환한다."""
    pattern = os.path.join(locata_root, "**", f"audio_array_{array_name}.wav")
    for wav in sorted(glob(pattern, recursive=True)):
        yield os.path.dirname(wav)


def to_dataframe(sample: dict) -> pd.DataFrame:
    """읽어온 샘플을 한 행짜리 DataFrame(sample_input 형식)으로 감싼다."""
    return pd.DataFrame({k: [v] for k, v in sample.items()})

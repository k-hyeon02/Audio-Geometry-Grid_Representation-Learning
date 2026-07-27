"""AGG-RL best.pt 평가: 논문 Table 3 프로토콜(MAE°, ACC10%).

평가셋 4종 (논문 Table 2):
    NAO robot    실측 LOCATA benchmark2 (12ch, seen)     -- pkl
    Eigenmike    실측 LOCATA eigenmike   (32ch, unseen)  -- pkl
    Dynamic-S    합성 dynamic 4-12ch (seen)              -- on-the-fly
    Dynamic-U    합성 dynamic 13-16ch (unseen)           -- on-the-fly

메트릭 (논문 A.11, Eq.24/25):
    - 프레임(VAD active)별로 iterative max-peak(A.8, 10° suppress)로 Tl개 DOA 추출
    - 3D angular distance(great-circle, Eq.20)로 min-permutation MAE
    - per-utterance 평균 후 데이터셋 평균, ACC10 = <10° 프레임 비율

사용법:
    python evaluate.py --checkpoint checkpoints/agg_rl_FM_paper/best.pt --mpe_type FM \
        --librispeech_val <test-clean> --ms_snsd_val <noise_test> \
        --nao_dir datasets/locata_eval/nao --eigenmike_dir datasets/locata_eval/eigenmike
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os
import pickle as pkl

import numpy as np
import torch

from data.dataset import SyntheticDOADataset, build_dataloader
from data.simulate import SimulationConfig
from model.main import AGG_RL


# ---------------------------------------------------------------------------
# 각도 / peak 유틸 (3D fibonacci grid)
# ---------------------------------------------------------------------------
def great_circle_deg(cart_a: np.ndarray, cart_b: np.ndarray) -> float:
    """두 단위벡터 사이 각거리(도). 논문 Eq.20: delta = arccos(<a,b>)."""
    a = cart_a / (np.linalg.norm(cart_a) + 1e-12)
    b = cart_b / (np.linalg.norm(cart_b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def iterative_max_peaks(spectrum: np.ndarray, cart: np.ndarray,
                        num_peaks: int, margin_deg: float = 10.0) -> np.ndarray:
    """논문 A.8: 최대값 선택 후 margin_deg 이내를 억제하며 num_peaks개 추출.

    spectrum: (D,) 프레임의 grid 확률
    cart:     (D, 3) fibonacci 단위벡터
    반환:     (num_peaks, 3) 선택된 후보의 카르테시안 좌표
    """
    scores = spectrum.copy()
    picked = []
    for _ in range(num_peaks):
        d = int(np.argmax(scores))
        picked.append(cart[d])
        # d와 margin_deg 이내인 후보 모두 억제 (Eq.20 각거리)
        cos_sim = cart @ (cart[d] / (np.linalg.norm(cart[d]) + 1e-12))
        ang = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
        scores[ang <= margin_deg] = -np.inf
    return np.asarray(picked, dtype=np.float64)


def min_permutation_mae(pred_cart: np.ndarray, tgt_cart: np.ndarray) -> float:
    """논문 Eq.24: 모든 순열 중 최소 평균 각오차(도)."""
    best = float("inf")
    n_pred = len(pred_cart)
    for perm in itertools.permutations(range(n_pred), len(tgt_cart)):
        errs = [great_circle_deg(pred_cart[p], tgt_cart[t])
                for t, p in enumerate(perm)]
        best = min(best, float(np.mean(errs)))
    return best


def sph_deg_to_cart(az_deg: float, el_deg: float) -> np.ndarray:
    """(azimuth°, elevation°) -> 단위 카르테시안. model.util.sph2cart와 동일 컨벤션."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.sin(el) * np.cos(az),
                     np.sin(el) * np.sin(az),
                     np.cos(el)], dtype=np.float64)


# ---------------------------------------------------------------------------
# 한 배치 평가 -> 프레임별 MAE 리스트
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_batch(model, device, input_audio, mic_coordinate, vad, spherical_position):
    """반환: 배치 내 샘플별 (active frame MAE 도) 리스트의 리스트 -> [[...], [...], ...].

    논문 A.11(Eq.24)의 utterance 단위 집계를 위해 샘플(=utterance)별로 분리해 반환한다.
    """
    input_audio = input_audio.to(device)
    mic_coordinate = mic_coordinate.to(device)
    vad = vad.to(device)

    out = model(input_audio, mic_coordinate)          # (B, O, D, L)
    out = out[:, -1].cpu().numpy()                    # 마지막 layer (B, D, L)
    # forward는 기본 fibonacci grid를 쓰므로 동일 grid를 재생성해 peak->각도 매핑
    from model.util import fibonacci_sphere
    cart = fibonacci_sphere(model.gridnet.fibonacci_size).cpu().numpy()  # (D, 3)

    # VAD를 output 프레임 길이에 맞춰 프레이밍
    vad_framed = model.input_feature_process.LNuDFT.get_vad_framed(vad).cpu().numpy()  # (B, n_spk, Lf)
    sph = spherical_position.cpu().numpy()             # (B, n_spk, 3, T)

    B, D, L = out.shape
    per_utt = []
    for b in range(B):
        frame_maes = []
        for l in range(L):
            active = np.flatnonzero(vad_framed[b, :, l] > 0)
            if active.size == 0:
                continue
            # ground-truth: active 화자의 (az, el) -> cart. 정적이므로 프레임 0 값 사용.
            tgt = np.stack([sph_deg_to_cart(sph[b, s, 0, 0], sph[b, s, 1, 0])
                            for s in active], axis=0)
            preds = iterative_max_peaks(out[b, :, l], cart, num_peaks=int(active.size))
            frame_maes.append(min_permutation_mae(preds, tgt))
        per_utt.append(frame_maes)
    return per_utt


def summarize(utterances: list[list[float]]) -> tuple[float, float, int]:
    """논문 A.11: utterance별 (active frame 평균)을 구한 뒤 utterance 평균.

    MAE   = mean_u( mean_frame( MAE ) )              (Eq.24)
    ACC10 = mean_u( frame들 중 <10deg 비율 )          (Eq.25, utterance 단위 평균)
    반환: (MAE deg, ACC10 %, 유효 utterance 수)
    """
    utt_mae, utt_acc = [], []
    for frames in utterances:
        if not frames:
            continue
        arr = np.asarray(frames, dtype=np.float64)
        utt_mae.append(float(arr.mean()))
        utt_acc.append(float((arr <= 10.0).mean() * 100.0))
    if not utt_mae:
        return float("nan"), float("nan"), 0
    return float(np.mean(utt_mae)), float(np.mean(utt_acc)), len(utt_mae)


# ---------------------------------------------------------------------------
# 실측(pkl) / 합성 로더
# ---------------------------------------------------------------------------
def _to_tensor(x, dtype):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


@torch.no_grad()
def eval_real_pkl(model, device, pkl_dir: str) -> tuple[float, float, int]:
    """LOCATA export .pkl 디렉터리 하나 평가 (recording=utterance 단위)."""
    files = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))
    utterances = []
    for fp in files:
        with open(fp, "rb") as f:
            df = pkl.load(f)
        row = df.iloc[0]
        input_audio = _to_tensor(row["input_audio"], torch.float32).unsqueeze(0)     # (1,C,T)
        mic = _to_tensor(row["mic_coordinate"], torch.float32).unsqueeze(0)          # (1,C,3)
        vad = _to_tensor(row["vad"], torch.float32).unsqueeze(0)                     # (1,n_spk,T)
        sph = _to_tensor(row["spherical_position"], torch.float64).unsqueeze(0)      # (1,n_spk,3,T)
        utterances.extend(eval_batch(model, device, input_audio, mic, vad, sph))
    return summarize(utterances)


@torch.no_grad()
def eval_synthetic(model, device, profile, channel_schedule, args) -> tuple[float, float, int]:
    """합성 프로필 하나 평가."""
    n_samples = len(channel_schedule) if channel_schedule is not None else args.fixed_suite_samples
    dataset = SyntheticDOADataset(
        librispeech_root=args.librispeech_val,
        ms_snsd_root=args.ms_snsd_val,
        num_samples=n_samples,
        profile=profile,
        batch_size=args.batch_size,
        seed=args.seed,
        simulation_config=SimulationConfig(),
        rotate_arrays=True,
        channel_schedule=channel_schedule,
    )
    dataset.set_epoch(0)
    loader = build_dataloader(dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=False)
    utterances = []
    for batch in loader:
        utterances.extend(eval_batch(model, device, batch["input_audio"],
                                     batch["mic_coordinate"], batch["vad"],
                                     batch["spherical_position"]))
    return summarize(utterances)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mpe_type", choices=["FM", "PM"], default="FM")
    ap.add_argument("--librispeech_val", required=True)
    ap.add_argument("--ms_snsd_val", required=True)
    ap.add_argument("--nao_dir", default="datasets/locata_eval/nao")
    ap.add_argument("--eigenmike_dir", default="datasets/locata_eval/eigenmike")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--samples_per_channel", type=int, default=300)  # 논문: 채널당 300 utterance
    ap.add_argument("--skip_synthetic", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AGG_RL(MPE_type=args.mpe_type).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    ep = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    print(f"checkpoint={args.checkpoint} (epoch={ep}, mpe={args.mpe_type})")
    print(f"device={device}\n")

    results = {}

    # --- 실측 ---
    if os.path.isdir(args.nao_dir) and glob.glob(os.path.join(args.nao_dir, "*.pkl")):
        results["NAO robot"] = eval_real_pkl(model, device, args.nao_dir)
    if os.path.isdir(args.eigenmike_dir) and glob.glob(os.path.join(args.eigenmike_dir, "*.pkl")):
        results["Eigenmike"] = eval_real_pkl(model, device, args.eigenmike_dir)

    # --- 합성 ---
    if not args.skip_synthetic:
        # Dynamic-S: 4-12ch, 채널당 samples_per_channel
        sched_s = tuple(c for c in range(4, 13) for _ in range(args.samples_per_channel))
        results["Dynamic-S"] = eval_synthetic(model, device, "dynamic4to12", sched_s, args)
        # Dynamic-U: 13-16ch (unseen), 채널당 samples_per_channel
        sched_u = tuple(c for c in range(13, 17) for _ in range(args.samples_per_channel))
        results["Dynamic-U"] = eval_synthetic(model, device, "dynamic13to16", sched_u, args)

    # --- 출력 ---
    print(f"{'dataset':<12} {'MAE(deg)':>10} {'ACC10(%)':>10} {'#frames':>10}")
    print("-" * 46)
    for name in ["NAO robot", "Eigenmike", "Dynamic-S", "Dynamic-U"]:
        if name in results:
            mae, acc10, n = results[name]
            print(f"{name:<12} {mae:>10.2f} {acc10:>10.2f} {n:>10d}")
    maes = [results[k][0] for k in results if not np.isnan(results[k][0])]
    accs = [results[k][1] for k in results if not np.isnan(results[k][1])]
    if maes:
        print("-" * 46)
        print(f"{'mean_mae':<12} {np.mean(maes):>10.2f} {np.mean(accs):>10.2f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import fftconvolve

import gpuRIR  # type: ignore


FS = 16_000
AUDIO_LEN = 4 * FS
N_SPK = 2
SPEED_OF_SOUND = 343.0

# 시뮬레이션 생성 환경
@dataclass(frozen=True)
class SimulationConfig:
    sample_rate: int = FS
    segment_seconds: float = 4.0
    max_speakers: int = N_SPK
    snr_db: tuple[float, float] = (-5.0, 30.0)
    utterance_sir_db: tuple[float, float] = (-5.0, 5.0)  # 여러 스피커가 있을 때 다른 스피커가 얼마나 간섭하는지
    noise_sir_db: tuple[float, float] = (-15.0, 15.0)       # 노이즈와 백색소음의 비율
    rt60_s: tuple[float, float] = (0.2, 1.3)              # Reverberation Time(RT60): 음이 1/1000로 감쇠하는 시간
    room_size_min_m: tuple[float, float, float] = (3.0, 3.0, 2.5)  # 최소 방 크기
    room_size_max_m: tuple[float, float, float] = (10.0, 8.0, 6.0) # 최대 방 크기
    source_distance_m: tuple[float, float] = (0.3, 2.5)  # 음원의 거리
    azimuth_deg: tuple[float, float] = (0.0, 360.0)
    elevation_deg: tuple[float, float] = (0.0, 180.0)  # elevation 범위 [0, 180]
    # AGG-RL(ICLR 2026) Appendix A.10: elevation은 vMF(mu=pi, kappa=2)에서 뽑은 뒤
    # 절반으로 줄여 수평면(~90도) 근처를 선호하게 만든다. 원래 GI-DOAEnet은 단순
    # uniform(30, 150)을 사용했으며, elevation_use_vmf=False로 그 동작을 복원할 수 있다.
    elevation_use_vmf: bool = True                      # True면 vMF(A.10), False면 uniform
    elevation_vmf_mu_deg: float = 180.0                 # vMF 평균 방향 mu (deg)
    elevation_vmf_kappa: float = 2.0                    # vMF 집중도 kappa
    min_speaker_gap_deg: float = 10.0                     # 스피커 간의 최소 간격
    min_wall_distance_m: float = 0.1                      # 벽에서의 최소 거리
    min_noise_distance_m: float = 2.5                     # 노이즈 소스는 마이크로부터 최소 2.5m 떨어뜨림
    rir_diffuse_attenuation_db: float = 12.0              # RIR 확산음 감쇠
    rir_end_attenuation_db: float = 40.0                  # RIR 끝부분 감쇠
    rir_convolution_device: str = "cpu"                   # RIR 적용 convolution 장치 ("cpu" or "cuda")

    @property
    def segment_samples(self) -> int:
        return int(self.sample_rate * self.segment_seconds)  # 16,000 * 4 = 64,000


# RMS(Root Mean Square) 계산: 신호의 크기 측정
def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64) + 1e-10))


# 신호를 정확한 길이로 조정
def _trim_or_pad(signal: np.ndarray, length: int) -> np.ndarray:
    signal = signal.astype(np.float32, copy=False)
    if signal.shape[0] >= length:  # 신호가 length보다 길면
        return signal[:length].copy()
    return np.pad(signal, (0, length - signal.shape[0])).astype(np.float32)


# 신호를 특정 dB 비율로 스케일링
def _scale_to_db(reference: np.ndarray, signal: np.ndarray, target_db: float) -> float:
    return _rms(reference) / (10.0 ** (target_db / 20.0) * (_rms(signal) + 1e-10))


# 신호 최댓값 기준으로 정규화, -0.95~0.95 범위로 조정되어 오버플러우 방지
def _normalize_peak(signal: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_value = np.max(np.abs(signal)) + 1e-10
    return (signal * (peak / max_value)).astype(np.float32)


def _sample_room_and_rt60(
    rng: np.random.Generator, config: SimulationConfig
) -> tuple[np.ndarray, float]:
    room = np.array(
        [
            rng.uniform(config.room_size_min_m[0], config.room_size_max_m[0]),  # 방의 가로(X) - min: 3.0m ~ max: 10.0m 범위의 균등 분포 
            rng.uniform(config.room_size_min_m[1], config.room_size_max_m[1]),  # 방의 세로(Y) - min: 3.0m ~ max: 8.0m 범위의 균등 분포 
            rng.uniform(config.room_size_min_m[2], config.room_size_max_m[2]),  # 방의 높이(Z) - min: 2.5m ~ max: 6.0m 범위의 균등 분포 
        ],
        dtype=np.float64,
    )
    rt60 = float(rng.uniform(*config.rt60_s))  # (0.2, 1.3)
    return room, rt60


# 마이크 배열의 중심 좌표
def _sample_array_center(
    room_size: np.ndarray, rng: np.random.Generator, config: SimulationConfig
) -> np.ndarray:
    z_low = min(max(0.5, config.min_wall_distance_m), room_size[2] - config.min_wall_distance_m)  # 마이크의 최소 높이
    z_high = max(z_low, min(room_size[2] - config.min_wall_distance_m, 2.0))                      # 마이크의 최대 높이
    return np.array(
        [
            rng.uniform(config.min_wall_distance_m, room_size[0] - config.min_wall_distance_m),  # 가로(X)
            rng.uniform(config.min_wall_distance_m, room_size[1] - config.min_wall_distance_m),  # 세로(Y)
            rng.uniform(z_low, z_high),                                                          # 높이(Z): 0.5 ~ 2.0m
        ],
        dtype=np.float64,
    )


def _room_corners(room_size: np.ndarray, config: SimulationConfig) -> np.ndarray:
    low = np.full(3, config.min_wall_distance_m, dtype=np.float64)
    high = room_size - config.min_wall_distance_m
    return np.array(
        [
            [low[0], low[1], low[2]],
            [low[0], low[1], high[2]],
            [low[0], high[1], low[2]],
            [low[0], high[1], high[2]],
            [high[0], low[1], low[2]],
            [high[0], low[1], high[2]],
            [high[0], high[1], low[2]],
            [high[0], high[1], high[2]],
        ],
        dtype=np.float64,
    )


# 구면 좌표를 직교 좌표로 변환
def _spherical_to_cartesian(
    distance_m: float, azimuth_deg: float, elevation_deg: float
) -> np.ndarray:
    # degree를 radian으로 변환
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return np.array(
        [
            distance_m * math.sin(elevation) * math.cos(azimuth),  # x좌표
            distance_m * math.sin(elevation) * math.sin(azimuth),  # y좌표
            distance_m * math.cos(elevation),                      # z좌표
        ],
        dtype=np.float64,
    )


# AGG-RL A.10 elevation 샘플러. 논문은 elevation을 "phi/2 (phi ~ vMF(mu=pi, kappa=2))
# 로 주어지며, 수평면 근처를 선호하되 범위를 [0, pi]로 제한한다"고 서술한다.
#
#   phi ~ vonMises(mu=pi, kappa)는 +-pi 근처에 집중되고, [0, 2pi)로 wrapping하면
#   pi 근처에 집중된다. 여기에 elev = phi/2를 적용하면 [0, pi](=[0, 180]도)로
#   매핑되며 90도(수평면) 근처에 집중된다. 이는 sample_input의 분포(평균 ~90도,
#   [0, 180] 전 구간, 수평면 대칭 peak)를 재현한다. elevation_use_vmf=False면
#   GI-DOAEnet의 uniform(30, 150) 동작을 쓴다.
def _sample_elevation_deg(
    rng: np.random.Generator, config: SimulationConfig
) -> float:
    if not config.elevation_use_vmf:
        return float(rng.uniform(*config.elevation_deg))      # uniform fallback
    mu = math.radians(config.elevation_vmf_mu_deg)            # 기본값 pi
    phi = float(rng.vonmises(mu, config.elevation_vmf_kappa))  # (-pi, pi], +-pi 근처 집중
    phi = phi % (2.0 * math.pi)                                # [0, 2pi), pi 근처 집중
    elev_deg = math.degrees(phi / 2.0)                         # [0, 180], 90도 근처 집중
    return float(np.clip(elev_deg, config.elevation_deg[0], config.elevation_deg[1]))


# 마이크 어레이 중심 기준으로 음원의 상대 위치 계산
def _sample_source_position(
    room_size: np.ndarray,
    array_center: np.ndarray,
    used_azimuths: Sequence[float],
    rng: np.random.Generator,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    '''
    - 스피커(음성): 마이크 가까이 (0.5m~)
        직접음이 강함
        위치 추정의 주요 대상
    '''
    for _ in range(1024):
        distance = float(rng.uniform(*config.source_distance_m))  # 마이크로부터의 거리: 0.3 ~ 2.5m
        azimuth = float(rng.uniform(*config.azimuth_deg))         # azimuth: 0 ~ 360
        elevation = _sample_elevation_deg(rng, config)            # elevation: vMF(A.10) 또는 uniform

        if any(
            min(abs(azimuth - prev), 360.0 - abs(azimuth - prev))
            < config.min_speaker_gap_deg  
            for prev in used_azimuths     # 현재 각도가 이전 각도에 대해 10도 미만이면
        ):
            continue  # 스피커 간 최소 10도 간격 필요 -> 10도 미만이면 다시 시도

        relative = _spherical_to_cartesian(distance, azimuth, elevation)  # 구면 좌표 -> 직교 좌표
        absolute = array_center + relative                                # 마이크 배열 중심으로부터의 절대 좌표
        
        # 유효성 검증: 스피커가 방 안에 있는가? -> 성공하면 절대좌표, 구면좌표 반환
        if np.all(absolute >= config.min_wall_distance_m) and np.all(absolute <= room_size - config.min_wall_distance_m): 
            polar = np.array([azimuth, elevation, distance], dtype=np.float32)
            return absolute.astype(np.float64), polar

    # 1024번의 루프가 모두 실패하면
    distance = float(rng.uniform(config.source_distance_m[0], min(2.5, config.source_distance_m[1])))  # 마이크로부터의 거리: 0.3~2.5m
    azimuth = float(rng.uniform(*config.azimuth_deg))
    elevation = _sample_elevation_deg(rng, config)
    relative = _spherical_to_cartesian(distance, azimuth, elevation)
    # 벽 범위 내로 클리핑
    absolute = np.clip(
        array_center + relative,
        config.min_wall_distance_m,
        room_size - config.min_wall_distance_m,
    )
    polar = np.array([azimuth, elevation, distance], dtype=np.float32)
    return absolute.astype(np.float64), polar


# 마이크 어레이 중심 기준으로 노이즈 음원의 상대 위치 계산
def _sample_noise_position(
    room_size: np.ndarray,
    array_center: np.ndarray,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> np.ndarray:
    '''
    - 노이즈(배경음): 마이크 멀리 (2.5m~)
        멀리서 오는 배경음 시뮬레이션
        스피커를 방해하지 않게
    '''
    low = np.full(3, config.min_wall_distance_m, dtype=np.float64)
    high = room_size - config.min_wall_distance_m
    for _ in range(1024):
        absolute = rng.uniform(low, high).astype(np.float64)
        if np.linalg.norm(absolute - array_center) >= config.min_noise_distance_m:
            return absolute

    corners = _room_corners(room_size, config)
    distances = np.linalg.norm(corners - array_center[None, :], axis=1)
    farthest_idx = int(np.argmax(distances))
    if distances[farthest_idx] >= config.min_noise_distance_m:
        return corners[farthest_idx]

    raise ValueError("Cannot place coherent noise at the required minimum distance.")


# RIR bank 생성
def _render_rir_bank(
    room_size: np.ndarray,
    source_positions: np.ndarray,
    mic_positions: np.ndarray,
    rt60: float,
    config: SimulationConfig,
) -> np.ndarray:
    # 벽 반사 계수
    beta = gpuRIR.beta_SabineEstimation(room_size, rt60)

    # 직접/초기 반사 이후에 확산음(diffuse field)으로 넘어간다고 볼 시간
    t_diff = float(gpuRIR.att2t_SabineEstimator(config.rir_diffuse_attenuation_db, rt60))
    
    # RIR를 어디까지 계산할지 정하는 최대 시간
    t_max = float(gpuRIR.att2t_SabineEstimator(config.rir_end_attenuation_db, rt60))

    # image source method에서 몇 차 반사까지 계산할지 정하는 값
    nb_img = gpuRIR.t2n(t_diff, room_size)

    # 실제 RIR 계산 호출
    return gpuRIR.simulateRIR(
        room_sz=room_size,
        beta=beta,
        pos_src=source_positions,
        pos_rcv=mic_positions,
        nb_img=nb_img,
        Tmax=t_max,
        fs=config.sample_rate,
        Tdiff=t_diff,
        spkr_pattern="omni",
        mic_pattern="omni",
    )


def _apply_rir_bank(
        signal: np.ndarray, 
        rir_bank: np.ndarray, 
        length: int,
        convolution_device: str = "cpu",
    ) -> np.ndarray:
    '''
    단일 채널 signal에 각 마이크용 RIR 적용하여 멀티채널 신호로
    각 마이크마다 조금씩 다르게 들리는 멀티채널 신호 반환
    '''
    if convolution_device == "cuda":
        try:
            import torch
            import torch.nn.functional as F

            if torch.cuda.is_available():
                signal_t = torch.from_numpy(signal).float().to("cuda").unsqueeze(0)  # (1, L)
                rir_t = torch.from_numpy(rir_bank).float().to("cuda")  # (C: 채널 수, R: RIR 길이)

                outputs = []
                for c in range(rir_t.shape[0]):
                    # 각 채널별 convolution
                    rir_c = rir_t[c].unsqueeze(0).unsqueeze(0)  # (R, , ) -> (1, 1, R)
                    out = F.conv1d(
                        signal_t.unsqueeze(1),  # (1, 1, L)
                        rir_c,                  # 필터로 사용할 현재 채널의 RIR
                        padding=rir_t.shape[1] - 1  # R-1 -> output length: L+R-1
                    ).squeeze().cpu().numpy()

                    # 원하는 length로 자르거나 패딩해서 저장
                    outputs.append(_trim_or_pad(out.astype(np.float32), length))

                return np.stack(outputs, axis=0)  # (C, length)
        except Exception:
            # GPU convolution이 실패하면 CPU convolution으로 fallback
            pass

    # 기본 경로: CPU convolution
    return np.stack(
        [
            _trim_or_pad(
                fftconvolve(signal, rir, mode="full").astype(np.float32),
                length,
            )
            for rir in rir_bank
        ],
        axis=0,
    )


def simulate_one_sample(
        speeches: Sequence[np.ndarray],
        coherent_noise: np.ndarray,
        mic_coords: np.ndarray,
        rng: np.random.Generator,
        config: SimulationConfig | None = None,
    ) -> dict[str, np.ndarray]:
    config = config or SimulationConfig()
    length = config.segment_samples
    num_channels = int(mic_coords.shape[0])  # mic_coors.shape = (C, 3) - C행 3열(x,y,z 좌표)
    max_speakers = config.max_speakers
    num_active_speakers = int(rng.integers(1, max_speakers + 1))  # 1~max_speaker 명

    for _ in range(1024):
        room_size, rt60 = _sample_room_and_rt60(rng, config)
        array_center = _sample_array_center(room_size, rng, config)
        # 상대좌표(mic_coords) + array_center -> 각 마이크의 절대좌표, 벽에 너무 붙지 않게 clipping
        mic_positions = np.clip(mic_coords.astype(np.float64) + array_center[None, :],
                                config.min_wall_distance_m,
                                room_size - config.min_wall_distance_m)

        source_positions = []  # 음원들의 절대 위치
        polar_positions = np.zeros((max_speakers, 3), dtype=np.float32)  # 화자 라벨 저장할 배열
        used_azimuths: list[float] = []  # 이미 사용한 azimuths

        for speaker_idx in range(num_active_speakers):
            # 화자의 절대좌표, 극좌표
            absolute, polar = _sample_source_position(
                room_size=room_size,
                array_center=array_center,
                used_azimuths=used_azimuths,
                rng=rng,
                config=config,
            )
            source_positions.append(absolute)
            polar_positions[speaker_idx] = polar
            used_azimuths.append(float(polar[0]))

        try:
            noise_position = _sample_noise_position(room_size, array_center, rng, config)
        except ValueError:
            continue
        break
    else:
        raise RuntimeError("Failed to sample a valid room geometry for coherent noise.")

    # RIR bank 생성
    rir_bank = _render_rir_bank(
        room_size=room_size,
        source_positions=np.vstack(source_positions + [noise_position]),
        mic_positions=mic_positions,
        rt60=rt60,
        config=config,
    )

    speech_signals = []  # 각 화자별 멀티채널 신호 저장
    for speaker_idx in range(max_speakers):
        if speaker_idx >= num_active_speakers:  # 비활성 화자면
            speech_signals.append(np.zeros((num_channels, length), dtype=np.float32))  # 0으로 채운 멀티채널 신호 삽입
            continue

        speech = _trim_or_pad(speeches[speaker_idx], length)  # 화자의 mono 음성을 length 길이로 맞춤
        speech_signals.append(
            _apply_rir_bank(
                speech,
                rir_bank[speaker_idx],
                length,
                convolution_device=config.rir_convolution_device,
            )
        )  # RIR bank 적용하여 멀티채널 신호로 변환

    coherent_noise = _trim_or_pad(coherent_noise, length)  # noise 길이를 length로 맞춤
    # noise에 RIR 적용하여 멀티채널 coherent noise 생성
    coherent_noise_mc = _apply_rir_bank(
        coherent_noise,
        rir_bank[num_active_speakers],
        length,
        convolution_device=config.rir_convolution_device,
    )
    # 채널별로 독립 백색잡음 생성
    white_noise = rng.standard_normal((num_channels, length)).astype(np.float32)

    mixed_speech = speech_signals[0].copy()
    for speaker_idx in range(1, num_active_speakers):  # 두 번쩨 active speaker 부터
        sir_db = float(rng.uniform(*config.utterance_sir_db))
        # 첫 번쩨 채널(화자)을 기준으로 현재 화자를 얼마나 키우거나 줄일지 계산
        scale = _scale_to_db(mixed_speech[0], speech_signals[speaker_idx][0], sir_db)
        # 스케일한 화자 신호를 현재 speech mixture에 더함
        mixed_speech += speech_signals[speaker_idx] * scale
    
    # coherent noise와 white noise 사이의 비율을 랜덤으로
    noise_sir_db = float(rng.uniform(*config.noise_sir_db))
    white_scale = _scale_to_db(coherent_noise_mc[0], white_noise[0], noise_sir_db)
    mixed_noise = coherent_noise_mc + white_noise * white_scale

    # speech 대 noise의 전체 SNR 목표값을 랜덤으로
    snr_db = float(rng.uniform(*config.snr_db))
    noise_scale = _scale_to_db(mixed_speech[0], mixed_noise[0], snr_db)  # noise 비율 설정
    mixture = _normalize_peak(mixed_speech + mixed_noise * noise_scale)  # speech + noise, peak normalize

    # AGG-RL inference.py는 (max_spk, 3, T) 형태의 spherical_position을 기대한다.
    # 모델이 LearnableNuDFT.get_trajectory_framed로 시간축에 대해 framing하여
    # 이동 음원까지 처리하기 때문이다. 여기서는 음원이 정적이므로 화자별
    # [az, el, dist]를 시간축으로 broadcast한다.
    spherical_position = np.broadcast_to(
        polar_positions[:, :, None], (max_speakers, 3, length)
    ).astype(np.float32)

    return {
        "input_audio": mixture.astype(np.float32),              # 최종 멀티채널 mixture (C, L)
        "mic_coordinate": mic_coords.astype(np.float32),        # 마이크 상대좌표 (C, 3)
        "polar_position": polar_positions.astype(np.float32),   # 각 화자의 [azimuth, elevation, distance] (GI-DOAEnet 호환)
        "spherical_position": spherical_position,               # (max_spk, 3, T) AGG-RL inference 포맷
        "n_spk": np.int64(num_active_speakers),                 # 실제 활성 화자 수
    }

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np  # 수치 계산 라이브러리
from scipy.signal import fftconvolve  # FFT 기반 고속 convolution

import gpuRIR  # type: ignore  # GPU RIR(Room Impulse Response) 시뮬레이션 라이브러리

# 음향 시뮬레이션의 기본 파라미터
FS = 16_000  # 샘플링 레이트: 16kHz
AUDIO_LEN = 4 * FS  # 오디오 길이: 4초 = 64,000 샘플
N_SPK = 2  # 최대 화자 수: 2명
SPEED_OF_SOUND = 343.0  # 음속: 실온 기준 343 m/s

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
    # sqrt(mean(signal^2) + eps): 신호의 에너지(크기)를 계산, 1e-10은 numerical stability 위한 작은값
    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64) + 1e-10))


# 신호를 정확한 길이로 조정
def _trim_or_pad(signal: np.ndarray, length: int) -> np.ndarray:
    signal = signal.astype(np.float32, copy=False)  # float32로 변환
    if signal.shape[0] >= length:  # 신호가 length보다 길면
        return signal[:length].copy()  # 처음 length 샘플만 자르기
    # 신호가 length보다 짧으면, 뒤에 0을 padding하여 length가 되도록 조정
    return np.pad(signal, (0, length - signal.shape[0])).astype(np.float32)


# 신호를 특정 dB 비율로 스케일링: ref 기준 signal과 target_dB만큼 차이나도록
# target_dB = 20log[{scale * rms(sig)}/rms(ref)] -> scale = rms(ref) * 10^(target_dB/20) / rms(sig)
def _scale_to_db(reference: np.ndarray, signal: np.ndarray, target_db: float) -> float:
    # reference에 대해 signal의 상대적 크기가 target_dB가 되도록 하는 scale factor 계산
    return _rms(reference) * (10.0 ** (target_db / 20.0)) / (_rms(signal) + 1e-10)


# 신호 최댓값 기준으로 정규화, -0.95~0.95 범위로 조정되어 오버플러우 방지
def _normalize_peak(signal: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_value = np.max(np.abs(signal)) + 1e-10  # 신호의 최댓값(절대값) 계산
    return (signal * (peak / max_value)).astype(np.float32)  # peak=0.95가 되도록 스케일링


def _sample_room_and_rt60(
    rng: np.random.Generator, config: SimulationConfig
) -> tuple[np.ndarray, float]:
    room = np.array(
        [
            rng.uniform(config.room_size_min_m[0], config.room_size_max_m[0]),  # 방의 가로(X): [3.0m, 10.0m]
            rng.uniform(config.room_size_min_m[1], config.room_size_max_m[1]),  # 방의 세로(Y): [3.0m, 8.0m]
            rng.uniform(config.room_size_min_m[2], config.room_size_max_m[2]),  # 방의 높이(Z): [2.5m, 6.0m]
        ],
        dtype=np.float64,
    )
    rt60 = float(rng.uniform(*config.rt60_s))  # Reverberation Time: 음이 1/1000로 감쇠하는 시간 [0.2s, 1.3s]
    return room, rt60


# 마이크 배열의 중심 좌표를 방 내에서 무작위 위치로 샘플링
def _sample_array_center(
    room_size: np.ndarray, rng: np.random.Generator, config: SimulationConfig
) -> np.ndarray:
    # 높이 범위 계산: 바닥/천장으로부터 최소거리 유지하면서 0.5~2.0m 범위
    z_low = min(max(0.5, config.min_wall_distance_m), room_size[2] - config.min_wall_distance_m)
    z_high = max(z_low, min(room_size[2] - config.min_wall_distance_m, 2.0))
    return np.array(
        [
            rng.uniform(config.min_wall_distance_m, room_size[0] - config.min_wall_distance_m),  # X좌표: 벽에서 최소거리
            rng.uniform(config.min_wall_distance_m, room_size[1] - config.min_wall_distance_m),  # Y좌표: 벽에서 최소거리
            rng.uniform(z_low, z_high),                                                          # Z좌표(높이): [z_low, z_high]
        ],
        dtype=np.float64,
    )


def _room_corners(room_size: np.ndarray, config: SimulationConfig) -> np.ndarray:
    # 방의 모서리 8개 좌표 계산 (벽에서 최소거리 내 범위)
    low = np.full(3, config.min_wall_distance_m, dtype=np.float64)  # 최소값 좌표
    high = room_size - config.min_wall_distance_m  # 최대값 좌표
    return np.array(
        [
            [low[0], low[1], low[2]],   # 모서리 1: (min_x, min_y, min_z)
            [low[0], low[1], high[2]],  # 모서리 2: (min_x, min_y, max_z)
            [low[0], high[1], low[2]],  # 모서리 3: (min_x, max_y, min_z)
            [low[0], high[1], high[2]], # 모서리 4: (min_x, max_y, max_z)
            [high[0], low[1], low[2]],  # 모서리 5: (max_x, min_y, min_z)
            [high[0], low[1], high[2]], # 모서리 6: (max_x, min_y, max_z)
            [high[0], high[1], low[2]], # 모서리 7: (max_x, max_y, min_z)
            [high[0], high[1], high[2]], # 모서리 8: (max_x, max_y, max_z)
        ],
        dtype=np.float64,
    )


# 구면 좌표(spherical)를 직교 좌표(cartesian)로 변환
# (r, azimuth, elevation) -> (x, y, z)
def _spherical_to_cartesian(
    distance_m: float, azimuth_deg: float, elevation_deg: float
) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)  # azimuth: degree -> radian
    elevation = math.radians(elevation_deg)  # elevation: degree -> radian
    return np.array(
        [
            distance_m * math.sin(elevation) * math.cos(azimuth),  # x = r*sin(el)*cos(az)
            distance_m * math.sin(elevation) * math.sin(azimuth),  # y = r*sin(el)*sin(az)
            distance_m * math.cos(elevation),                      # z = r*cos(el)
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
        # 균등분포: 논문의 GI-DOAEnet 동작 복원
        return float(rng.uniform(*config.elevation_deg))
    # von Mises 분포 사용: 수평면(90도) 근처에 집중
    mu = math.radians(config.elevation_vmf_mu_deg)  # mu=180도(π) -> radian 변환
    phi = float(rng.vonmises(mu, config.elevation_vmf_kappa))  # von Mises(μ=π, κ=2): (-π, π] 범위의 각도 샘플
    phi = phi % (2.0 * math.pi)  # (-π, π] -> [0, 2π): 음수를 양수로 정규화
    elev_deg = math.degrees(phi / 2.0)  # [0, 2π) -> [0, π] radian -> [0, 180] degree (90도 근처 집중)
    # 설정된 범위 내로 클리핑
    return float(np.clip(elev_deg, config.elevation_deg[0], config.elevation_deg[1]))


# 마이크 어레이 중심 기준으로 음원의 상대 위치 계산
def _sample_source_position(
    room_size: np.ndarray,
    array_center: np.ndarray,
    used_azimuths: Sequence[float],
    rng: np.random.Generator,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    # 스피커 위치 샘플링: 유효한 위치가 나올때까지 최대 1024번 시도
    for _ in range(1024):
        # 구면 좌표로 음원 위치 샘플링
        distance = float(rng.uniform(*config.source_distance_m))  # 거리: [0.3m, 2.5m]
        azimuth = float(rng.uniform(*config.azimuth_deg))  # 방위각: [0도, 360도]
        elevation = _sample_elevation_deg(rng, config)  # 고도각: vMF 또는 균등분포

        # 스피커 간 최소 간격(10도) 확인
        if any(
            min(abs(azimuth - prev), 360.0 - abs(azimuth - prev))  # 원형 각도 거리 계산
            < config.min_speaker_gap_deg  # 최소 간격과 비교
            for prev in used_azimuths  # 기존 스피커 각도들과 비교
        ):
            continue  # 간격 미충족 -> 다시 샘플링

        # 구면 좌표 -> 직교 좌표 변환
        relative = _spherical_to_cartesian(distance, azimuth, elevation)  # 마이크 중심 기준 상대 좌표
        absolute = array_center + relative  # 방 전체 기준 절대 좌표

        # 유효성 검증: 스피커가 방 내 벽으로부터 최소거리를 유지하는가?
        if np.all(absolute >= config.min_wall_distance_m) and np.all(absolute <= room_size - config.min_wall_distance_m):
            # 조건 만족 -> 절대좌표와 극좌표 반환
            polar = np.array([azimuth, elevation, distance], dtype=np.float32)
            return absolute.astype(np.float64), polar

    # Fallback: 1024번 시도 후 실패시, 벽 범위 내로 강제로 클리핑
    distance = float(rng.uniform(config.source_distance_m[0], min(2.5, config.source_distance_m[1])))
    azimuth = float(rng.uniform(*config.azimuth_deg))
    elevation = _sample_elevation_deg(rng, config)
    relative = _spherical_to_cartesian(distance, azimuth, elevation)
    # 벽 범위 내로 강제 클리핑
    absolute = np.clip(
        array_center + relative,
        config.min_wall_distance_m,
        room_size - config.min_wall_distance_m,
    )
    polar = np.array([azimuth, elevation, distance], dtype=np.float32)
    return absolute.astype(np.float64), polar


# 마이크 어레이 중심 기준으로 노이즈 음원의 위치 계산
# 노이즈는 스피커와 달리 마이크로부터 최소 2.5m 이상 떨어져야함 (배경음 시뮬레이션)
def _sample_noise_position(
    room_size: np.ndarray,
    array_center: np.ndarray,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> np.ndarray:
    # 방 내 유효 범위 정의 (벽으로부터 최소거리)
    low = np.full(3, config.min_wall_distance_m, dtype=np.float64)
    high = room_size - config.min_wall_distance_m

    # 유효한 노이즈 위치 샘플링 (최대 1024번 시도)
    for _ in range(1024):
        absolute = rng.uniform(low, high).astype(np.float64)  # 방 내 균등분포로 샘플링
        # 마이크로부터의 거리가 최소거리(2.5m) 이상인가?
        if np.linalg.norm(absolute - array_center) >= config.min_noise_distance_m:
            return absolute  # 조건 만족 -> 위치 반환

    # Fallback: 1024번 실패시, 방의 가장 먼 모서리로 지정
    corners = _room_corners(room_size, config)  # 8개 모서리 좌표
    distances = np.linalg.norm(corners - array_center[None, :], axis=1)  # 각 모서리까지의 거리
    farthest_idx = int(np.argmax(distances))  # 가장 먼 모서리 인덱스
    if distances[farthest_idx] >= config.min_noise_distance_m:
        return corners[farthest_idx]  # 가장 먼 모서리 위치 반환

    # 예외: 아무 위치도 최소거리 조건을 만족하지 못하면 에러
    raise ValueError("Cannot place coherent noise at the required minimum distance.")


# RIR(Room Impulse Response) bank 생성
# 각 음원(스피커, 노이즈)에 대한 모든 마이크의 RIR을 계산
def _render_rir_bank(
    room_size: np.ndarray,
    source_positions: np.ndarray,
    mic_positions: np.ndarray,
    rt60: float,
    config: SimulationConfig,
) -> np.ndarray:
    # Sabine 이론으로 벽 반사 계수(beta) 계산: RT60 기반
    beta = gpuRIR.beta_SabineEstimation(room_size, rt60)

    # 초기 반사에서 확산음(diffuse field)으로 전환되는 시간
    # 이 시점 이후로는 확산음만 계산하여 계산량 감소
    t_diff = float(gpuRIR.att2t_SabineEstimator(config.rir_diffuse_attenuation_db, rt60))

    # RIR 계산 최대 시간: 이 이후는 무시할 정도로 약해짐
    t_max = float(gpuRIR.att2t_SabineEstimator(config.rir_end_attenuation_db, rt60))

    # Image source method: t_diff 시점까지 몇 차 반사까지 계산할지 결정
    nb_img = gpuRIR.t2n(t_diff, room_size)

    # gpuRIR로 실제 RIR 계산 (Image source + 확산음 혼합)
    return gpuRIR.simulateRIR(
        room_sz=room_size,  # 방의 크기 [X, Y, Z]
        beta=beta,  # 벽 반사 계수
        pos_src=source_positions,  # 음원 위치 (num_sources, 3)
        pos_rcv=mic_positions,  # 마이크 위치 (num_mics, 3)
        nb_img=nb_img,  # Image source 계산 깊이
        Tmax=t_max,  # RIR 최대 시간
        fs=config.sample_rate,  # 샘플링 레이트
        Tdiff=t_diff,  # 확산음으로 전환되는 시간
        spkr_pattern="omni",  # 무지향성 스피커
        mic_pattern="omni",  # 무지향성 마이크
    )


def _apply_rir_bank(
        signal: np.ndarray,
        rir_bank: np.ndarray,
        length: int,
        convolution_device: str = "cpu",
    ) -> np.ndarray:
    # 단일 채널 signal에 각 마이크용 RIR을 convolution하여 멀티채널 신호로 변환
    # 각 마이크마다 조금씩 다른 RIR이 적용되므로 서로 다른 음향 특성 가짐
    if convolution_device == "cuda":
        try:
            import torch
            import torch.nn.functional as F

            if torch.cuda.is_available():
                # signal: (L,) -> (1, L) 텐서로 변환, GPU로 전송
                signal_t = torch.from_numpy(signal).float().to("cuda").unsqueeze(0)  # (1, L)
                # rir_bank: (C, R) 텐서로 변환, GPU로 전송 (C: 채널 수, R: RIR 길이)
                rir_t = torch.from_numpy(rir_bank).float().to("cuda")

                outputs = []
                for c in range(rir_t.shape[0]):
                    # 채널 c의 RIR 가져오기
                    rir_c = rir_t[c].unsqueeze(0).unsqueeze(0)  # (R,) -> (1, 1, R)
                    # Conv1d로 convolution 수행: signal_t(1,1,L) * rir_c(1,1,R) -> (1,1,L+R-1)
                    out = F.conv1d(
                        signal_t.unsqueeze(1),  # (1, L) -> (1, 1, L)
                        rir_c,  # 필터: (1, 1, R)
                        padding=rir_t.shape[1] - 1  # R-1 padding으로 output length = L+R-1
                    ).squeeze().cpu().numpy()  # (1,1,L+R-1) -> (L+R-1,) -> numpy로 변환

                    # 결과를 원하는 length로 trim 또는 pad
                    outputs.append(_trim_or_pad(out.astype(np.float32), length))

                return np.stack(outputs, axis=0)  # (C, length) 형태로 스택
        except Exception:
            # GPU convolution 실패 시 CPU로 fallback
            pass

    # CPU convolution (기본 경로)
    return np.stack(
        [
            _trim_or_pad(
                # FFT 기반 고속 convolution으로 각 채널별 RIR 적용
                fftconvolve(signal, rir, mode="full").astype(np.float32),
                length,  # 결과를 원하는 길이로 조정
            )
            for rir in rir_bank  # rir_bank의 각 채널(행)에 대해 반복
        ],
        axis=0,  # 채널 축으로 스택: (C, length)
    )


def simulate_one_sample(
        speeches: Sequence[np.ndarray],
        coherent_noise: np.ndarray,
        mic_coords: np.ndarray,
        rng: np.random.Generator,
        config: SimulationConfig | None = None,
    ) -> dict[str, np.ndarray]:
    # 입력: speeches (활성 화자별 mono 음성), coherent_noise (배경음), mic_coords (마이크 배열 상대좌표)
    # 출력: 멀티채널 mixture와 라벨 정보
    config = config or SimulationConfig()  # 기본 설정 사용
    length = config.segment_samples  # 오디오 길이: 64,000 샘플
    num_channels = int(mic_coords.shape[0])  # 마이크 채널 수 (mic_coords shape: (C, 3))
    max_speakers = config.max_speakers  # 최대 화자 수: 2
    num_active_speakers = int(rng.integers(1, max_speakers + 1))  # 활성 화자 수: 1 또는 2

    # 유효한 방 구성 샘플링 (최대 1024번 시도)
    for _ in range(1024):
        # 방의 크기와 reverb time 샘플링
        room_size, rt60 = _sample_room_and_rt60(rng, config)
        # 마이크 배열 중심 위치 샘플링
        array_center = _sample_array_center(room_size, rng, config)
        # 각 마이크의 절대 좌표: 상대좌표(mic_coords) + array_center, 벽 범위 내 클리핑
        mic_positions = np.clip(mic_coords.astype(np.float64) + array_center[None, :],
                                config.min_wall_distance_m,
                                room_size - config.min_wall_distance_m)

        source_positions = []  # 활성 화자들의 절대 위치 저장
        polar_positions = np.zeros((max_speakers, 3), dtype=np.float32)  # 모든 화자의 극좌표 [az, el, dist]
        used_azimuths: list[float] = []  # 이미 사용한 azimuth 각도

        # 각 활성 화자의 위치 샘플링
        for speaker_idx in range(num_active_speakers):
            # 화자의 절대좌표(cartesian)와 극좌표(spherical) 샘플링
            absolute, polar = _sample_source_position(
                room_size=room_size,
                array_center=array_center,
                used_azimuths=used_azimuths,
                rng=rng,
                config=config,
            )
            source_positions.append(absolute)  # 절대좌표 저장
            polar_positions[speaker_idx] = polar  # 극좌표 저장
            used_azimuths.append(float(polar[0]))  # azimuth 기록

        # 노이즈 위치 샘플링 (마이크로부터 최소 2.5m 떨어짐)
        try:
            noise_position = _sample_noise_position(room_size, array_center, rng, config)
        except ValueError:
            continue  # 노이즈 위치를 찾지 못하면 전체 방 구성 재샘플링
        break  # 유효한 구성 찾음
    else:
        raise RuntimeError("Failed to sample a valid room geometry for coherent noise.")

    # RIR bank 생성: 활성 화자 + 노이즈 위치에 대한 모든 마이크의 RIR
    rir_bank = _render_rir_bank(
        room_size=room_size,
        source_positions=np.vstack(source_positions + [noise_position]),  # (num_sources, 3)
        mic_positions=mic_positions,  # (num_channels, 3)
        rt60=rt60,
        config=config,
    )  # rir_bank shape: (num_sources, num_channels, rir_length)

    # 각 화자의 mono 음성을 RIR로 멀티채널 변환
    speech_signals = []  # 화자별 멀티채널 신호
    for speaker_idx in range(max_speakers):
        if speaker_idx >= num_active_speakers:
            # 비활성 화자: 0으로 채운 더미 신호 (max_speakers와의 shape 맞춤)
            speech_signals.append(np.zeros((num_channels, length), dtype=np.float32))
            continue

        # 활성 화자: mono 음성에 RIR 적용하여 멀티채널 신호 생성
        speech = _trim_or_pad(speeches[speaker_idx], length)  # mono 음성을 정확한 길이로 조정
        speech_signals.append(
            _apply_rir_bank(
                speech,  # (L,)
                rir_bank[speaker_idx],  # (num_channels, rir_length)
                length,
                convolution_device=config.rir_convolution_device,
            )
        )  # (num_channels, length)

    # 배경음(coherent noise)을 멀티채널로 변환
    coherent_noise = _trim_or_pad(coherent_noise, length)  # 길이 조정
    coherent_noise_mc = _apply_rir_bank(
        coherent_noise,
        rir_bank[num_active_speakers],  # 노이즈의 RIR (rir_bank의 마지막 행)
        length,
        convolution_device=config.rir_convolution_device,
    )  # (num_channels, length)

    # 채널별 독립 백색 잡음 생성
    white_noise = rng.standard_normal((num_channels, length)).astype(np.float32)

    # 여러 화자의 음성 혼합
    mixed_speech = speech_signals[0].copy()  # 첫 번째 화자부터 시작
    for speaker_idx in range(1, num_active_speakers):
        # 화자 간 간섭 (SIR) 설정: 첫 화자 기준 현재 화자를 얼마나 키우거나 줄일지
        sir_db = float(rng.uniform(*config.utterance_sir_db))  # SIR: [-5, 5] dB
        # 스케일 계산: 첫 화자(기준)과의 dB 차이가 sir_db가 되도록
        scale = _scale_to_db(mixed_speech[0], speech_signals[speaker_idx][0], sir_db)
        # 스케일된 화자 신호 혼합
        mixed_speech += speech_signals[speaker_idx] * scale

    # 두 가지 노이즈 혼합: coherent noise + white noise
    noise_sir_db = float(rng.uniform(*config.noise_sir_db))  # coherent noise 대 white noise 비율
    white_scale = _scale_to_db(coherent_noise_mc[0], white_noise[0], noise_sir_db)
    mixed_noise = coherent_noise_mc + white_noise * white_scale  # (num_channels, length)

    # 음성 대 노이즈의 전체 SNR 설정
    snr_db = float(rng.uniform(*config.snr_db))  # SNR: [-5, 30] dB
    noise_scale = _scale_to_db(mixed_speech[0], mixed_noise[0], snr_db)
    # 최종 혼합: speech + noise, peak normalization으로 클리핑 방지
    mixture = _normalize_peak(mixed_speech + mixed_noise * noise_scale)  # (num_channels, length)

    # 극좌표를 (max_speakers, 3, length) 형태로 확장
    # AGG-RL 모델은 시간축 framing을 지원하므로, 정적 음원은 모든 시간에 동일한 위치
    spherical_position = np.broadcast_to(
        polar_positions[:, :, None],  # (max_speakers, 3, 1)
        (max_speakers, 3, length)  # (max_speakers, 3, length)로 broadcast
    ).astype(np.float32)

    return {
        # 시뮬레이션 결과 반환
        "input_audio": mixture.astype(np.float32),  # 최종 멀티채널 mixture: (num_channels, length)
        "mic_coordinate": mic_coords.astype(np.float32),  # 마이크 배열 상대좌표: (num_channels, 3)
        "polar_position": polar_positions.astype(np.float32),  # 화자의 극좌표: (max_speakers, 3) = [az, el, dist]
        "spherical_position": spherical_position,  # 시간축 확장된 극좌표: (max_speakers, 3, length)
        "n_spk": np.int64(num_active_speakers),  # 실제 활성 화자 수 (1 또는 2)
    }

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from glob import glob
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf
import torch
import webrtcvad
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset, Sampler

from .mic_arrays import get_fixed_array, random_rotate, sample_dynamic_array
from .simulate import N_SPK, SimulationConfig, simulate_one_sample


# MSGL stage는 AGG-RL Table 6을 따른다: stage 1 = 고정 4cm 정사면체(4ch),
# stage 2 = dynamic(4ch), stage 3 = dynamic(4-12ch).
PROFILE_SPECS = {
    "stage1": {"array_type": "tetrahedron", "channel_range": (4, 4)},
    "stage2": {"array_type": "dynamic", "channel_range": (4, 4)},
    "stage3": {"array_type": "dynamic", "channel_range": (4, 12)},
    "tetrahedron": {"array_type": "tetrahedron", "channel_range": (4, 4)},
    "nao4": {"array_type": "nao4", "channel_range": (4, 4)},
    "nao12": {"array_type": "nao12", "channel_range": (12, 12)},
    "dynamic4": {"array_type": "dynamic", "channel_range": (4, 4)},
    "dynamic4to12": {"array_type": "dynamic", "channel_range": (4, 12)},
}

WEBRTC_VAD_FRAME_MS = 30
WEBRTC_VAD_MODE = 3


def _discover_audio_files(root: str, patterns: Iterable[str]) -> list[str]:
    files: list[str] = []
    # "*.wav", "*.flac" 같은 pattern
    for pattern in patterns:
        # root 아래 모든 하위 폴더를 재귀적으로 뒤져 현재 패턴과 맞는 파일을 추가
        files.extend(glob(os.path.join(root, "**", pattern), recursive=True))
    return sorted(set(files))   # 중복 경로를 제거하고 정렬된 목록으로 반환



def _load_audio_mono(path: str) -> tuple[np.ndarray, int]:
    # 파일을 읽고, 가능한 한 원본 차원을 유지한 채 샘플레이트도 함께 받는다.
    audio, sample_rate = sf.read(path, always_2d=False)
    # 입력이 2차원이라면 멀티채널 오디오이므로 채널 평균으로 모노로
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # shape: (샘플수, )
    # 오디오는 float32로, 샘플레이트는 int로 통일해 (배열, 샘플레이트) 튜플 반환
    return audio.astype(np.float32), int(sample_rate)


def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    # 원본 샘플레이트와 목표 샘플레이트가 같으면 형식만 맞춰 그대로 반환
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    # 업샘플/다운샘플 비율을 단순화하기 위해 최대공약수
    gcd = np.gcd(src_sr, dst_sr)
    up = dst_sr // gcd      # 목표 샘플레이트 쪽 분자를 업샘플 비율로 사용
    down = src_sr // gcd    # 원본 샘플레이트 쪽 분모를 다운샘플 비율로 사용
    # polyphase resampling을 적용하고 결과를 float32로
    return resample_poly(audio, up=up, down=down).astype(np.float32)


def _crop_or_pad(
    audio: np.ndarray,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # 오디오 길이가 목표 길이 이상이면 랜덤한 시작점에서 크롭
    if audio.shape[0] >= length:
        # 잘라낼 수 있는 시작 인덱스를 균등하게 하나 샘플링
        start = int(rng.integers(0, audio.shape[0] - length + 1))
        # 선택된 구간만 반환하고 dtype은 float32로 통일
        return audio[start : start + length].astype(np.float32, copy=False)
    # 오디오가 더 짧으면 뒤쪽을 0으로 패딩
    return np.pad(audio, (0, length - audio.shape[0])).astype(np.float32)


def _sample_vad_mask(
    audio: np.ndarray,            # 깨끗한 단일 화자 음성 (shape: (샘플수,), float32, -1~1 범위)
    sample_rate: int,             # 음성의 샘플레이트 (WebRTC VAD는 8k/16k/32k/48k만 허용)
    frame_ms: int = WEBRTC_VAD_FRAME_MS,  # VAD가 한 번에 판정하는 프레임 길이 (ms 단위, 기본 30ms)
    mode: int = WEBRTC_VAD_MODE,  # VAD 공격성 (0=관대, 3=가장 엄격하게 비음성 제거)
) -> np.ndarray:
    # ms 단위 프레임 길이를 샘플 개수로 환산 (예: 16000Hz * 30ms / 1000 = 480 샘플)
    frame_length = int(sample_rate * frame_ms / 1000)
    # 프레임 길이가 0 이하이면 이후 슬라이싱이 불가능하므로 즉시 막는다.
    if frame_length <= 0:
        raise ValueError("frame_ms must produce a positive frame length.")
    # 빈 오디오면 마스크도 빈 배열로 그대로 반환 (이후 연산에서 0 나눗셈 등 방지)
    if audio.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    # 지정한 공격성으로 WebRTC VAD 판정기 생성
    vad = webrtcvad.Vad(mode)
    # float(-1~1)을 WebRTC가 요구하는 16-bit PCM 정수로 변환 (clip으로 범위 초과 방지)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    # 전체 길이를 frame_length의 배수로 맞추기 위해 필요한 뒤쪽 패딩 양 계산
    pad = (-pcm.shape[0]) % frame_length
    # 패딩이 필요하면 0으로 뒤를 채우고, 아니면 원본 그대로 사용
    padded = np.pad(pcm, (0, pad)) if pad else pcm
    # 프레임별 음성 여부(1.0/0.0)를 기록할 샘플 단위 마스크를 0으로 초기화
    active = np.zeros(padded.shape[0], dtype=np.float32)

    try:
        # 오디오를 frame_length 간격으로 잘라 프레임 단위로 순회
        for start in range(0, padded.shape[0], frame_length):
            # 현재 프레임의 끝 인덱스
            stop = start + frame_length
            # 이 프레임이 음성으로 판정되면 (바이트로 넘겨야 함)
            if vad.is_speech(padded[start:stop].tobytes(), sample_rate):
                # 해당 프레임 구간 전체를 1.0(발화)으로 표시
                active[start:stop] = 1.0
    except ValueError as exc:
        # sample_rate가 WebRTC 미지원 값이면 is_speech가 ValueError → 원인을 명시해 재발생
        raise ValueError(
            "WebRTC VAD failed while labeling a clean speech segment. "
            f"sample_rate={sample_rate}, frame_ms={frame_ms}"
        ) from exc

    # 패딩으로 늘어난 뒷부분을 잘라내 원래 오디오 길이에 맞춘 마스크 반환
    return active[: audio.shape[0]]


class SyntheticDOADataset(Dataset):
    # LibriSpeech와 MS-SNSD를 조합해 합성 DOA 학습 샘플을 만들어주는 Dataset
    def __init__(
        self,
        librispeech_root: str,          # 음성 파일 위치
        ms_snsd_root: str,              # 잡음 파일 위치
        num_samples: int,               # 반환할 샘플 수
        profile: str = "stage1",        # 학습 단계
        batch_size: int = 16,
        seed: int = 0,
        simulation_config: SimulationConfig | None = None,  # RIR 시뮬레이션 조건 설정
        rotate_arrays: bool = True,                         # 마이크 배열 회전 설정
        channel_schedule: Sequence[int] | None = None,      # 채널 수 설정 옵션
    ) -> None:
        # 전달받은 프로필 이름이 미리 정의된 설정에 없으면 바로 막는다.
        if profile not in PROFILE_SPECS:
            raise ValueError(f"Unknown profile '{profile}'.")

        self.librispeech_root = librispeech_root    # 음성 데이터 경로
        self.ms_snsd_root = ms_snsd_root            # 잡음 데이터 경로
        self.num_samples = int(num_samples)         # 생성할 전체 샘플 수
        self.profile = profile                      # 사용할 프로필 이름
        self.batch_size = int(batch_size)           # 배치 크기 (채널 수를 배치 단위로 묶을 때 사용)
        self.seed = int(seed)                       # 재현 가능한 샘플링을 위한 기본 시드를 저장
        self.rotate_arrays = rotate_arrays          # 마이크 배열을 랜덤 회전시킬지

        # 시뮬레이션 설정이 없으면 기본 SimulationConfig를 생성해 사용
        self.simulation_config = simulation_config or SimulationConfig()

        # 채널 스케줄(샘플별 채널 수를 직접 지정한 목록)이 주어졌다면 int32 배열로 변환
        if channel_schedule is not None:
            self.channel_schedule = np.asarray(channel_schedule, dtype=np.int32)
        # 안 주어졌으면 None → 나중에 프로필 범위에서 랜덤 배정하겠다는 의미
        else:
            self.channel_schedule = None

        # 채널 스케줄이 있으면 그 길이가 곧 샘플 수이므로 num_samples를 거기에 맞춤
        if self.channel_schedule is not None:
            self.num_samples = int(self.channel_schedule.shape[0])

        self._epoch = 0   # 현재 에폭 번호 (set_epoch로 갱신되며 시드 계산에 사용)

        # 오디오 데이터 수집
        self.speech_files = _discover_audio_files(librispeech_root, ("*.flac", "*.wav"))
        self.noise_files = _discover_audio_files(ms_snsd_root, ("*.wav", "*.flac"))

        if not self.speech_files:
            raise FileNotFoundError(f"No speech files were found under '{librispeech_root}'.")
        if not self.noise_files:
            raise FileNotFoundError(f"No noise files were found under '{ms_snsd_root}'.")

        # 현재 시드 기준으로 각 샘플의 채널 수 배정
        self.channel_counts = self._assign_channel_counts(self.seed)

    def __len__(self) -> int:
        # Dataset이 제공하는 총 샘플 수를 반환
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        # 외부 학습 루프가 넘긴 현재 에폭 번호를 저장
        self._epoch = int(epoch)
        # 에폭이 바뀌면 채널 수 배정도 새 시드로 다시 생성
        self.channel_counts = self._assign_channel_counts(self.seed + self._epoch)

    def set_profile(self, profile: str) -> None:
        # MSGL stage 전환 시 마이크 배열 프로필(stage1/2/3)을 바꾼다 (AGG-RL Table 6).
        if profile not in PROFILE_SPECS:
            raise ValueError(f"Unknown profile '{profile}'.")
        self.profile = profile
        # 프로필이 달라지면 채널 범위가 달라지므로 채널 수를 다시 배정
        self.channel_counts = self._assign_channel_counts(self.seed + self._epoch)

    def _assign_channel_counts(self, seed: int) -> np.ndarray:
        # 채널 스케줄을 직접 입력했다면 그것을 그대로 복사해 사용
        if self.channel_schedule is not None:
            return self.channel_schedule.copy()

        # 채널 수 배정을 위한 난수 생성기
        rng = np.random.default_rng(seed)
        # 현재 프로필이 허용하는 최소/최대 채널 수 범위
        min_channels, max_channels = PROFILE_SPECS[self.profile]["channel_range"]
        # 최소와 최대가 같으면 모든 샘플의 채널 수가 고정이므로 동일 값으로 채움
        if min_channels == max_channels:
            return np.full(self.num_samples, min_channels, dtype=np.int32)

        # 전체 샘플 수를 배치 크기로 나눠 필요한 배치 개수를 계산
        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        # 각 배치마다 사용할 채널 수를 허용 범위에서 하나씩 랜덤 추출
        batch_ch_counts = rng.integers(min_channels, max_channels + 1, size=num_batches)
        # 배치별 채널 수를 샘플 단위로 펼친 뒤 전체 샘플 수만큼만 잘라 반환 -> 배치 내 샘플의 채널 수 같도록
        return np.repeat(batch_ch_counts, self.batch_size)[: self.num_samples].astype(np.int32)

    def _sample_mic_coordinates(
        self, num_channels: int, rng: np.random.Generator
    ) -> np.ndarray:
        # 현재 프로필에서 사용할 마이크 배열 타입
        array_type = PROFILE_SPECS[self.profile]["array_type"]
        # 동적 배열 프로필이면 채널 수에 맞춰 매번 새로운 배열 샘플링
        if array_type == "dynamic":
            coords = sample_dynamic_array(num_channels, rng=rng)
        # 고정 배열 프로필이면 미리 정의된 배열 좌표
        else:
            coords = get_fixed_array(array_type)

        # 옵션이 켜져 있으면 배열 전체를 랜덤하게 회전
        if self.rotate_arrays:
            coords = random_rotate(coords, rng)
        # 이후 torch 변환 전에 float32 numpy 배열로 맞춰 반환
        return coords.astype(np.float32)

    def _sample_audio(
        self, file_path: str, rng: np.random.Generator
    ) -> np.ndarray:
        # 지정된 파일에서 모노 오디오와 샘플레이트를 읽음
        audio, sample_rate = _load_audio_mono(file_path)
        # 시뮬레이션 설정의 샘플레이트에 맞게 리샘플링
        audio = _resample_audio(audio, sample_rate, self.simulation_config.sample_rate)
        # 필요한 세그먼트 길이에 맞게 랜덤 크롭 또는 zero-padding을 적용
        return _crop_or_pad(audio, self.simulation_config.segment_samples, rng)

    def _sample_speech(
        self, file_path: str, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        # 음성 파일을 모노/리샘플/고정 길이로 전처리
        speech = self._sample_audio(file_path, rng)
        # 논문 서술에 맞춰, RIR/혼합 전에 clean utterance에서 WebRTC VAD를 계산한다.
        vad = _sample_vad_mask(
            speech,
            sample_rate=self.simulation_config.sample_rate,
        )
        return speech, vad.astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # 시드, 에폭, 인덱스를 조합해 샘플별로 재현 가능한 난수 시드 생성 - 같은 에폭의 같은 인덱스면 같은 샘플이 나옴
        seed = (self.seed * 1_000_003 + self._epoch * self.num_samples + index) & 0xFFFFFFFF
        # 방금 만든 시드로 난수 생성기 생성
        rng = np.random.default_rng(seed)
        # 현재 샘플의 채널 수
        num_channels = int(self.channel_counts[index])

        # 화자 수 N_SPK만큼 서로 다른 음성 파일 인덱스를 중복 없이 랜덤 선택
        speech_indices = rng.choice(len(self.speech_files), size=N_SPK, replace=False)
        # 선택된 각 음성 파일을 전처리 -> 각 원소가 (음성 배열, VAD 마스크) 튜플인 리스트
        speech_tracks = [self._sample_speech(self.speech_files[i], rng) for i in speech_indices]
        # 튜플에서 음성 배열만 뽑아 화자별 음성 리스트로 분리
        speeches = [speech for speech, _ in speech_tracks]
        # 튜플에서 VAD 마스크만 뽑아 화자 축으로 쌓음 -> shape: (N_SPK, 샘플수)
        vad = np.stack([track_vad for _, track_vad in speech_tracks], axis=0).astype(np.float32)
        # 잡음 파일 목록에서 하나의 인덱스를 랜덤 선택
        noise_index = int(rng.integers(0, len(self.noise_files)))
        # 선택된 잡음 파일도 전처리
        noise = self._sample_audio(self.noise_files[noise_index], rng)
        # 현재 샘플의 채널 수에 맞는 마이크 좌표 샘플링
        mic_coordinates = self._sample_mic_coordinates(num_channels, rng)

        # 음성, 잡음, 마이크 배열을 조합해 음향 시뮬레이션 수행 -> 다채널 학습 샘플 생성
        sample = simulate_one_sample(
            speeches=speeches,
            coherent_noise=noise,
            mic_coords=mic_coordinates,
            rng=rng,
            config=self.simulation_config,
        )
        # 실제 활성 화자 수(n_spk)보다 뒤쪽 화자 슬롯은 비어 있으므로 VAD를 0으로 비움
        vad[int(sample["n_spk"]) :] = 0.0

        # numpy 결과를 torch 텐서로 바꿔 학습 루프가 바로 쓸 수 있게 반환
        return {
            # 모델 입력이 되는 다채널 오디오를 텐서로 변환
            "input_audio": torch.from_numpy(sample["input_audio"]),
            # clean utterance에서 얻은 WebRTC VAD 기반 화자별 샘플 단위 발화 마스크
            "vad": torch.from_numpy(vad),
            # 마이크 배열 좌표도 텐서로 변환
            "mic_coordinate": torch.from_numpy(sample["mic_coordinate"]),
            # AGG-RL inference 포맷의 화자 위치 정답 - 시간축까지 포함 (max_spk, 3, T)
            "spherical_position": torch.from_numpy(sample["spherical_position"]),
            # 화자 위치의 polar 좌표 정답도 텐서로 변환 - GI-DOAEnet 호환 정적 좌표 (max_spk, 3)
            "polar_position": torch.from_numpy(sample["polar_position"]),
            # 화자 수는 정수 라벨이므로 long tensor로 감싼다.
            "n_spk": torch.tensor(sample["n_spk"], dtype=torch.long),
        }


class ChannelGroupBatchSampler(Sampler[list[int]]):
    # 채널 수가 같은 샘플끼리만 한 배치에 묶어주는 Sampler
    def __init__(
        self,
        channel_counts: Sequence[int],  # 각 샘플의 채널 수 목록
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        # 채널 수 목록을 int32 numpy 배열로 보관
        self.channel_counts = np.asarray(channel_counts, dtype=np.int32)
        # 배치 크기를 정수로 저장
        self.batch_size = int(batch_size)
        # 배치 순서를 섞을지 여부를 저장
        self.shuffle = shuffle
        # 마지막 미완성 배치를 버릴지 여부를 저장
        self.drop_last = drop_last
        # 현재 채널 수 분포 기준으로 배치 인덱스 묶음을 미리 계산
        self._batches = self._build_batches()

    def _build_batches(self) -> list[list[int]]:
        # 채널 수별로 샘플 인덱스를 모을 딕셔너리
        groups: dict[int, list[int]] = {}   # ex) {4: [0,1,5,7], 8: [2,3,4], 12: [6,8]}
        '''
        키가 없으면: groups[key] = []를 먼저 만들고, 거기에 index 추가
        키가 이미 있으면: 기존 리스트를 그대로 가져와서, 거기에 index 추가
        '''
        for index, count in enumerate(self.channel_counts.tolist()):
            groups.setdefault(int(count), []).append(index)

        # 최종적으로 반환할 배치 목록을 담을 리스트
        batches: list[list[int]] = []
        # 같은 채널 수를 가진 인덱스 그룹마다 배치를 생성
        for indices in groups.values():
            # 현재 그룹을 batch_size 간격으로 잘라 배치
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                # 마지막 배치가 모자라고 drop_last가 켜져 있으면 버린다.
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                # 유효한 배치만 결과 목록에 추가
                batches.append(batch)
        # 완성된 배치 목록을 반환
        return batches

    def __iter__(self):
        # 원본 배치 목록을 건드리지 않도록 얕은 복사본
        batches = self._batches.copy()
        # 셔플 옵션이 켜져 있으면 배치 순서를 무작위로 섞는다.
        if self.shuffle:
            np.random.shuffle(batches)
        # 준비된 배치를 하나씩 순서대로 내보낸다.
        yield from batches

    def __len__(self) -> int:
        # 현재 샘플러가 생성한 배치 개수를 반환
        return len(self._batches)


def build_dataloader(
    dataset: SyntheticDOADataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    prefetch_factor: int = 1,
) -> DataLoader:
    # 같은 채널 수끼리 묶는 배치 샘플러를 먼저 생성
    sampler = ChannelGroupBatchSampler(
        channel_counts=dataset.channel_counts,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
    # DataLoader에 넘길 공통 옵션을 딕셔너리로 정리
    dataloader_kwargs: dict = {
        # 일반 sampler 대신 batch_sampler를 써서 배치 구성을 직접 제어
        "batch_sampler": sampler,
        # 데이터 로딩에 사용할 worker 프로세스 수를 지정
        "num_workers": num_workers,
        # GPU 전송을 빠르게 하기 위해 pinned memory를 켠다.
        "pin_memory": True,
        # worker가 있으면 epoch 사이에도 프로세스를 유지
        "persistent_workers": False,
    }
    # worker를 실제로 쓸 때만 멀티프로세싱 관련 옵션을 추가
    if num_workers > 0:
        # spawn 컨텍스트를 사용해 worker 프로세스를 안전하게 시작
        dataloader_kwargs["multiprocessing_context"] = mp.get_context("spawn")
        # gpuRIR가 worker 안에서 CUDA를 쓰므로 과도한 선읽기는 GPU OOM을 유발할 수 있다.
        dataloader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    # 준비된 옵션을 풀어서(**) DataLoader를 생성해 반환
    return DataLoader(dataset, **dataloader_kwargs)

# -------------------------------------- 검증용 ------------------------------------------------


def _run_dataset_smoke_test(args: argparse.Namespace) -> None:
    # 명령줄 인자로 받은 설정으로 데이터셋을 실제 생성 (파일 탐색/채널 배정까지 수행됨)
    dataset = SyntheticDOADataset(
        librispeech_root=args.librispeech_root,
        ms_snsd_root=args.ms_snsd_root,
        num_samples=args.num_samples,
        profile=args.profile,
        batch_size=args.batch_size,
        seed=args.seed,
        rotate_arrays=not args.no_rotate_arrays,  # --no-rotate-arrays 주면 회전 끔
    )
    # 에폭을 지정해 채널 배정 시드를 고정 (재현 가능한 검사용)
    dataset.set_epoch(args.epoch)
    # 채널 수가 같은 샘플끼리 묶는 DataLoader 생성
    loader = build_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # 배정된 채널 수의 고유값과 각 빈도를 세어 분포를 출력하기 위함
    unique_counts, counts_freq = np.unique(dataset.channel_counts, return_counts=True)
    print("=== SyntheticDOADataset smoke test ===")
    print(f"librispeech_root={dataset.librispeech_root}")
    print(f"ms_snsd_root={dataset.ms_snsd_root}")
    print(f"profile={dataset.profile}")
    print(f"num_samples={len(dataset)}")
    print(f"batch_size={args.batch_size}")
    print(f"num_workers={args.num_workers}")
    print(f"speech_files={len(dataset.speech_files)}")
    print(f"noise_files={len(dataset.noise_files)}")
    print(
        "channel_counts_distribution="
        + ", ".join(
            f"{int(ch)}ch:{int(freq)}"
            for ch, freq in zip(unique_counts.tolist(), counts_freq.tolist())
        )
    )

    inspected_batches = 0   # 실제로 검사한 배치 개수 카운터
    # DataLoader를 돌며 배치를 하나씩 꺼내 형태를 검사
    for batch_idx, batch in enumerate(loader):
        # 배치 딕셔너리에서 각 텐서를 꺼냄
        input_audio = batch["input_audio"]
        vad = batch["vad"]
        mic_coordinate = batch["mic_coordinate"]
        spherical_position = batch["spherical_position"]
        polar_position = batch["polar_position"]
        n_spk = batch["n_spk"]

        # 각 텐서의 shape/dtype을 출력해 육안 점검
        print(f"batch[{batch_idx}]")
        print(
            "  input_audio="
            f"shape={tuple(input_audio.shape)} dtype={input_audio.dtype}"
        )
        print(
            "  vad="
            f"shape={tuple(vad.shape)} dtype={vad.dtype}"
        )
        print(
            "  mic_coordinate="
            f"shape={tuple(mic_coordinate.shape)} dtype={mic_coordinate.dtype}"
        )
        print(
            "  spherical_position="
            f"shape={tuple(spherical_position.shape)} dtype={spherical_position.dtype}"
        )
        print(
            "  polar_position="
            f"shape={tuple(polar_position.shape)} dtype={polar_position.dtype}"
        )
        print(f"  n_spk=shape={tuple(n_spk.shape)} values={n_spk.tolist()}")

        # 배치 첫 번째 축(샘플 수)이 요청한 batch_size와 일치하는지 확인
        if input_audio.shape[0] != args.batch_size:
            raise RuntimeError(
                f"Unexpected batch size {input_audio.shape[0]} != {args.batch_size}"
            )
        # 오디오 채널 수와 마이크 좌표 개수가 일치하는지 확인 (둘 다 두 번째 축이 채널)
        if input_audio.shape[1] != mic_coordinate.shape[1]:
            raise RuntimeError(
                "Channel mismatch between input_audio and mic_coordinate: "
                f"{input_audio.shape[1]} != {mic_coordinate.shape[1]}"
            )
        # VAD의 배치 축이 입력 오디오의 배치 축과 같은지 확인
        if vad.shape[0] != input_audio.shape[0]:
            raise RuntimeError(
                f"Batch dimension mismatch between vad and input_audio: {vad.shape[0]} != {input_audio.shape[0]}"
            )
        # polar 위치 라벨의 배치 축이 입력 오디오의 배치 축과 같은지 확인
        if polar_position.shape[0] != input_audio.shape[0]:
            raise RuntimeError(
                "Batch dimension mismatch between polar_position and input_audio: "
                f"{polar_position.shape[0]} != {input_audio.shape[0]}"
            )

        inspected_batches += 1   # 이 배치를 정상 검사 완료
        # 요청한 검사 배치 수에 도달하면 루프 종료
        if inspected_batches >= args.num_batches:
            break

    # 한 배치도 못 만들었다면 샘플 수/배치 크기 설정이 잘못된 것
    if inspected_batches == 0:
        raise RuntimeError(
            "No batches were produced. Increase --num-samples or lower --batch-size."
        )

    print(f"Smoke test passed after inspecting {inspected_batches} batch(es).")


# 이 파일을 직접 실행했을 때만(import될 때는 제외) 아래 검증 코드가 동작
if __name__ == "__main__":
    # 이 파일 기준 상위 폴더 = 프로젝트 루트의 절대경로 (실행 위치와 무관하게 고정)
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    # 명령줄 인자를 해석할 파서 생성
    parser = argparse.ArgumentParser(
        description="Smoke test SyntheticDOADataset and its DataLoader."
    )
    # 음성(LibriSpeech) 폴더 경로. 미지정 시 프로젝트 내 기본 위치 사용
    parser.add_argument(
        "--librispeech-root",
        default=os.path.join(_repo_root, "datasets", "librispeech", "LibriSpeech", "test-clean"),
        help="Path to the LibriSpeech split to sample speech utterances from.",
    )
    # 잡음(MS-SNSD) 폴더 경로. 미지정 시 프로젝트 내 기본 위치 사용
    parser.add_argument(
        "--ms-snsd-root",
        default=os.path.join(_repo_root, "datasets", "ms-snsd", "MS-SNSD", "noise_test"),
        help="Path to the MS-SNSD split to sample noise clips from.",
    )
    parser.add_argument("--num-samples", type=int, default=4)   # 생성할 샘플 수
    # 사용할 프로필 (PROFILE_SPECS의 키들 중에서만 선택 가능)
    parser.add_argument("--profile", choices=sorted(PROFILE_SPECS.keys()), default="stage1")
    parser.add_argument("--batch-size", type=int, default=2)    # 배치 크기
    parser.add_argument("--num-workers", type=int, default=0)   # DataLoader worker 프로세스 수
    parser.add_argument("--seed", type=int, default=0)          # 난수 시드
    parser.add_argument("--epoch", type=int, default=0)         # 검사에 쓸 에폭 번호
    # 종료 전 몇 개의 배치를 돌며 검사할지
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="How many batches to iterate through before exiting.",
    )
    # 플래그형 옵션: 주면 True가 되어 마이크 배열 회전을 끔 (결정적 검사용)
    parser.add_argument(
        "--no-rotate-arrays",
        action="store_true",
        help="Disable random microphone-array rotation for deterministic inspection.",
    )
    # 인자를 실제로 해석해 검증 함수에 넘김
    _run_dataset_smoke_test(parser.parse_args())

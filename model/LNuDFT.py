# torch-stft와 ConferencingSpeech2021 구현을 참고한 LNuDFT 레이어

import math  # π 등 수학 상수

import numpy as np  # 윈도우와 초기 주파수 간격 생성을 위한 NumPy
import torch  # 텐서 연산과 자동미분을 위한 PyTorch
import torch.nn as nn  # 신경망 모듈과 학습 파라미터
import torch.nn.functional as F  # padding과 1차원 convolution 함수
from scipy.signal import get_window  # 지정된 시간 영역 윈도우 생성 함수


EPSILON = torch.finfo(torch.float32).eps  # 0으로 나누거나 sqrt(0)가 되는 것을 막는 작은 값


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    """일반 DFT/역 DFT용 convolution 커널 초기화 함수(현재 LearnableNuDFT에서는 미사용)."""
    del win_inc  # 과거 API 호환성을 위해 유지하는 미사용 인자
    if win_type == "None" or win_type is None:  # 무윈도우 설정 여부
        window = np.ones(win_len)  # 모든 표본을 통과시키는 직사각 윈도우
    else:  # 지정 윈도우 사용 분기
        window = get_window(win_type, win_len, fftbins=True)  # FFT 기준 시간 윈도우

    fourier_basis = np.fft.rfft(np.eye(fft_len))[:win_len]  # 필요한 시간축 구간의 실수 입력 DFT 기저, np.eye: 단위 행렬 생성
    real_kernel = np.real(fourier_basis)  # 복소 DFT 기저의 실수부: (win_len, F)
    imag_kernel = np.imag(fourier_basis)  # 복소 DFT 기저의 허수부: (win_len, F)

    # 주파수(F) 축으로 실수·허수부 결합 후 전치로 출력 채널 축 생성 -> (2F, win_len) = (출력채널, 커널길이)
    kernel = np.concatenate([real_kernel, imag_kernel], axis=1).T
    if invers:  # 역변환용 커널 요청 여부
        kernel = np.linalg.pinv(kernel).T  # 의사역행렬 기반 역변환 근사 커널: (win_len, 2F).T -> (2F, win_len)

    kernel = kernel * window  # 시간 표본별 분석 윈도우 적용
    kernel = kernel[:, None, :]  # conv1d용 입력 채널 축 추가
    kernel_tensor = torch.from_numpy(kernel.astype(np.float32))  # float32 PyTorch 커널 텐서
    window_tensor = torch.from_numpy(window[None, :, None].astype(np.float32))  # 호환 형태의 윈도우 텐서
    return kernel_tensor, window_tensor  # DFT 커널과 윈도우 텐서 반환


class LearnableNuDFT(nn.Module):
    """학습 가능한 비균일 주파수 bin 기반 1차원 DFT 분석 레이어."""

    """초기화:
    eps_start ~ eps_end의 K개 위치 생성
    → logit 변환
    → 0 ~ win_len/2 범위로 정규화
    → 인접 위치 차이 계산
    → K-1개의 a_k_parameter 초기값
    → 누적합 + nu_0
    → K개의 비균일 주파수 위치 nu_k"""

    def __init__(
        self,
        win_len: int,  # 프레임 표본 수이자 DFT 정규화 길이
        hop_len: int,  # 인접 프레임 간 hop(보폭) 길이
        K: int,  # 출력 비균일 주파수 bin 개수
        vad_threshold=2 / 3,  # 음성 구간 판정용 평균 VAD 임계값
        win_type: str = "hann",  # 분석용 시간 영역 윈도우 종류
        fs: float = 16000.0,  # 입력 샘플링 주파수 메타데이터
        a_k_trainable: bool = False,  # 주파수 bin 간격 학습 여부
        eps_0=0.01,  # 간격 클리핑 하한
        eps_1=100,  # 간격 클리핑 상한
        nu_0: float = 0.0,  # 첫 비균일 주파수 bin의 시작 위치
        eps_start: float = 0.15,  # 초기 bin 분포의 시작 비율
        eps_end: float = 0.95,  # 초기 bin 분포의 끝 비율
    ):
        super().__init__()  # PyTorch 모듈 내부 상태 초기화

        self.win_len = win_len  # 프레임/윈도우 길이
        self.stride = hop_len  # convolution 시간축 보폭
        self.win_type = win_type  # 윈도우 종류
        self.fs = fs  # 샘플링 주파수
        self.eps_0 = eps_0  # 학습 간격 최소 허용값
        self.eps_1 = eps_1  # 학습 간격 최대 허용값
        self.nu_0 = nu_0  # 첫 번째 주파수 bin 위치
        self.K = K  # 전체 주파수 bin 개수

        """*nu 자체를 파라미터화 하지 않고 간격을 학습 가능한 파라미터로 설정한 이유:
        주파수 bin의 순서와 범위를 안정적으로 유지하기 위해서
        - 직접 bin 위치를 학습하면 각 값이 독립적으로 업데이트 되므로 bin 순서가 뒤집히거나, 두 bin이 같은 위치로 몰릴 수 있음
        - 간격을 학습하면 주파수 bin의 순서 유지/중복 방지할 수 있고, 마지막 bin의 Nyquist 조건을 보장할 수 있음
        """
        a_k_list = self.a_k_init(eps_start, eps_end)  # K-1개 양의 초기 bin 간격
        self.a_k_parameter = nn.Parameter(  # bin 간격 학습 파라미터 등록
            torch.tensor(a_k_list, dtype=torch.float32),  # NumPy 초기값의 float32 텐서 변환
            requires_grad=a_k_trainable,  # 설정에 따른 간격 gradient 활성화
        )

        if self.win_type is None or self.win_type.lower() == "none":  # 무윈도우 설정 분기
            window_np = np.ones(self.win_len, dtype=np.float32)  # 직사각 윈도우
        else:  # 일반 분석 윈도우 사용 분기
            window_np = get_window(self.win_type, self.win_len, fftbins=True).astype(np.float32)  # 선택 윈도우 생성
        self.register_buffer(
            "window", torch.from_numpy(window_np)
        )  # register_buffer(name, tensor): 학습하지 않는 텐서를 모델 구성요소로 등록하는 함수 (device 이동·저장 대상)

        n = torch.arange(self.win_len, dtype=torch.float32)  # 기저 함수용 시간 표본 인덱스 0부터 N-1까지
        self.register_buffer("n", n)  # 비학습 시간 인덱스 buffer 등록

        self.vad_threshold = vad_threshold  # VAD 프레임화용 임계값
        vad_kernel = torch.ones((1, 1, self.win_len), dtype=torch.float32) / self.win_len  # 프레임 평균용 이동평균 커널
        self.register_buffer("vad_kernel", vad_kernel)  # VAD/궤적 프레임화용 커널 buffer -> self.vad_kernel=vad_kernel

    def get_vad_framed(self, vad):
        """표본 단위 VAD의 LNuDFT 프레임 단위 이진 VAD 변환."""
        batch_size, speaker_count, sample_count = vad.shape  # 입력 배치·화자·시간 축 길이 (B: 배치 크기, P: 화자 수, L: 표본 수)
        vad = vad.view(batch_size * speaker_count, 1, sample_count)  # 화자별 독립 conv1d 입력 형태: (Batch, Channel, Time)
        pad_size = (self.stride - sample_count) % self.stride  # 마지막 프레임이 hop 간격에 맞도록 오른쪽 padding 길이
        vad = F.pad(vad, [0, pad_size])  # 마지막 불완전 프레임의 0 padding
        vad = F.conv1d(vad, self.vad_kernel, stride=self.stride)  # 프레임별 평균 VAD
        # 이진 VAD와 원래 축 구조 복원 (B, P, L), ge: greater than or equal -> Boolean tensor
        vad = (vad.view(batch_size, speaker_count, -1).ge(self.vad_threshold).long()) # vad >= vad_threshold면 1.0, 아니면 0.0
        return vad  # (B, 화자 수, 프레임 수) 이진 VAD 반환

    def a_k_init(self, eps_start, eps_end):
        """logit 공간에서 고르게 배치한 초기 비균일 bin 간격 생성."""
        numbering = np.linspace(eps_start, eps_end, self.K, dtype=np.float32)  # (0, 1) 범위 K개 초기 위치
        logit_init = np.log(numbering / (1.0 - numbering))  # 양 끝 간격 확대와 중앙 bin 밀도 증가를 위한 logit 공간 변환
        logit_init = logit_init - logit_init[0]  # 첫 위치를 0으로 맞추는 원점 이동
        logit_init = logit_init / logit_init[-1] * (self.win_len / 2.0)  # 마지막 위치를 Nyquist bin으로 맞추는 정규화
        a_k_list = logit_init[1:] - logit_init[:-1]  # 누적 위치 차이 기반 K-1개 bin 간격
        return a_k_list.tolist()  # nn.Parameter용 Python 리스트 반환 -> bin 간격을 학습 파라미터로

    def _compute_nu(self):
        """양수 간격 누적 및 Nyquist 범위 정규화 기반 비균일 bin 인덱스 nu_k 계산."""
        temp_a_k_data = self.a_k_parameter.data.clone()  # 범위 제약 확인용 파라미터 복사본
        temp_a_k_data = torch.clamp(temp_a_k_data, min=self.eps_0, max=self.eps_1)  # eps_0 ~ eps_1 양의 허용 범위 내 간격 제한
        temp_a_k_sum = torch.sum(temp_a_k_data)  # 전체 bin 간격 합

        if temp_a_k_sum > self.win_len / 2.0:  # 누적 간격의 Nyquist 범위 초과 여부
            self.a_k_parameter.data = temp_a_k_data / temp_a_k_sum * (self.win_len / 2.0)  # 합을 Nyquist bin으로 맞추는 축소

        a_k_cumsum = torch.cumsum(self.a_k_parameter, dim=0)  # K-1개 간격 누적합 기반 각 bin 위치
        nu = a_k_cumsum + self.nu_0  # 시작 bin 위치 반영
        nu_0_tensor = torch.tensor([self.nu_0], device=nu.device, dtype=nu.dtype)  # 현재 device/dtype 기반 시작 bin 텐서
        nu = torch.cat([nu_0_tensor, nu], dim=0)  # 시작 bin 결합 기반 총 K개 위치
        return nu  # (K,) 비균일 DFT bin 인덱스 반환

    def get_trajectory_framed(self, trajectory):
        """표본 단위 3차원 궤적의 LNuDFT 프레임 격자 평균 궤적 변환."""
        batch_size, speaker_count, coordinate_count, sample_count = trajectory.shape  # 입력 궤적의 축별 크기 (B, P, 3: xyz 좌표, L)
        trajectory = trajectory.contiguous()  # view 전 연속 메모리 배치 보장
        trajectory = trajectory.view(batch_size * speaker_count * coordinate_count, 1, sample_count)  # 좌표별 독립 conv1d 입력 형태: (Batch, Channel, Time)
        pad_size = (self.stride - sample_count) % self.stride  # 마지막 프레임 정렬용 오른쪽 padding 길이
        trajectory = F.pad(trajectory, [0, pad_size], mode="replicate")  # 마지막 좌표 복제 기반 프레임 padding -> (B*P*3, 1, L_padded)
        trajectory = F.conv1d(trajectory, self.vad_kernel, stride=self.stride)  # 프레임별 평균 좌표
        trajectory = trajectory.view(batch_size, speaker_count, coordinate_count, -1)  # 배치·화자·좌표 축 복원
        return trajectory  # (B, P, 3, T: 프레임 수) 평균 궤적 반환

    def _build_kernels(self):
        """nu_k 기반 exp(-j·2pi·n·nu_k/N) 기저의 conv1d 커널 생성."""
        nu = self._compute_nu()  # 현재 학습 비균일 bin 위치 (K, )
        # bin·시간 표본별 복소 지수 위상: (K, 1) * (1, win_len) = (K, win_len)
        angle = -2 * math.pi * (nu[:, None] * self.n[None, :] / self.win_len)  
        cos_part = torch.cos(angle)  # 실수부용 cosine 기저
        sin_part = torch.sin(angle)  # 허수부용 sine 기저
        real_kernels = cos_part.unsqueeze(1)  # (K, 1, win_len) conv1d 실수부 커널 형태
        imag_kernels = sin_part.unsqueeze(1)  # (K, 1, win_len) conv1d 허수부 커널 형태

        window = self.window.view(1, 1, -1)  # 모든 주파수 커널 broadcast용 윈도우 형태
        real_kernels = real_kernels * window  # 실수부 기저의 분석 윈도우 적용
        imag_kernels = imag_kernels * window  # 허수부 기저의 분석 윈도우 적용
        kernels = torch.cat([real_kernels, imag_kernels], dim=0)  # 출력 채널 방향 실수·허수부 결합
        return kernels  # (2K, 1, win_len) NUDFT convolution 커널 반환

    def forward(self, inputs: torch.Tensor, cplx: bool = True):
        """입력 (B, C, L)의 프레임별 비균일 DFT 실·허수부 또는 크기·위상 변환."""
        batch_size, channel_count, sample_count = inputs.shape  # 입력 배치·채널·시간 축 길이
        conv1d_kernel = self._build_kernels()  # 현재 bin 위치 대응 NUDFT 기저 커널
        pad_size = (self.stride - sample_count % self.stride) % self.stride  # hop 단위 프레임 정렬용 오른쪽 padding 길이
        x = inputs.view(batch_size * channel_count, 1, sample_count)  # 채널별 독립 conv1d 입력 형태
        x = F.pad(x, [0, pad_size])  # 마지막 불완전 프레임의 0 padding
        outputs = F.conv1d(x, conv1d_kernel.to(x.dtype), stride=self.stride)  # 프레임과 모든 NUDFT 기저의 내적
        outputs = outputs.view(batch_size, channel_count, 2 * self.K, -1)  # 배치·채널 축 구조 복원
        real, imag = torch.chunk(outputs, 2, dim=2)  # 앞 K개 실수부와 뒤 K개 허수부 분리

        if cplx:  # 복소 표현 요청 여부
            return real, imag  # 실수부와 허수부 반환

        mags = torch.sqrt(real**2 + imag**2 + EPSILON)  # 수치적으로 안전한 복소 스펙트럼 크기
        phase = torch.atan2(imag, real + EPSILON)  # 수치적으로 안전한 복소 스펙트럼 위상
        return mags, phase  # 크기와 위상 반환

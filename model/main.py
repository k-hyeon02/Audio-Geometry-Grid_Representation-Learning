import torch
from .input_feature_process import input_feature_process
from .LNuDFT import LearnableNuDFT
from .Channel_invariant_feature_extractor import Channel_invariant_feature_extractor
from .Microphone_positional_encoding import MicrophonePositionalEncoding
from .SpatioTemporal_block import SpatioTemporal_block
from .Representation_mapping import Representation_mapping_block
from .Gridnet import Gridnet_list
from .util import target_spatial_spectrum

class AGG_RL(torch.nn.Module):
    def __init__(self, MPE_type):
        super(AGG_RL, self).__init__()

        win_len = 512
        feature_size = 128
        ref_channel = 0
        MPE_type = MPE_type
        representation_size = 256
        fibonacci_size = 2048
        self.ref_channel=ref_channel

        self.input_feature_process=input_feature_process(win_len=win_len, ref_channel=ref_channel)

        self.MPE = MicrophonePositionalEncoding(feature=feature_size,
                                                MPE_type=MPE_type,
                                                alpha = 7,
                                                beta = 4,
                                                ref_channel=ref_channel,)

        self.CIFE=Channel_invariant_feature_extractor(init_feature = win_len+2,
                                                      feature = feature_size, 
                                                      kernel_size = 3, 
                                                      padding = 1, 
                                                      stride = 1, 
                                                      dilation_rate = 2, 
                                                      num_blocks = 4,)

        self.MPE = MicrophonePositionalEncoding(feature=feature_size,
                                                MPE_type=MPE_type,
                                                alpha = 7,
                                                beta = 4)        

        self.STDPBs=SpatioTemporal_block(n_heads=16,
                                        num_blocks=4,
                                        feature_size=feature_size,
                                        rnn_layers=2,)        

        self.RMBs=Representation_mapping_block(feature = feature_size,
                                            representation_size = representation_size,
                                            num_blocks = 3,
                                            kernel_size = 3, 
                                            dilation_rate = 2)      

        self.gridnet = Gridnet_list(num_depths=3, num_blocks=3, representation_size=representation_size, fibonacci_size =fibonacci_size, xi=1.0)

        # len(self.gammas)는 x_out의 DS(depth-stage) 축과 같아야 함 (= self.gridnet의 num_blocks).
        # target_spatial_spectrum이 gammas 길이만큼 G축을 만들고, 이 축이 x_out의 DS와 shape-matching되어
        # loss.py의 spatial_spectrum_loss에서 element-wise로 비교되기 때문 (둘 다 DS로 통일해 표기).
        self.gammas = [5.0, 5.0, 5.0]

    def ref_channel_modify_to_center(self, x, mic_coordinate):

        center = mic_coordinate.mean(dim=1, keepdim=True)  # (B, 1, 3)

        nearest_channel = torch.argmin(torch.norm(mic_coordinate - center, dim=-1), dim=1)

        x_new = x.clone()
        mic_coordinate_new = mic_coordinate.clone()
        rc = self.ref_channel 

        # Gather values to swap
        batch_indices = torch.arange(x.shape[0], device=x.device)
        # Swap x
        x_new[batch_indices, rc], x_new[batch_indices, nearest_channel] = \
            x[batch_indices, nearest_channel].clone(), x[batch_indices, rc].clone()
        # Swap mic_pos
        mic_coordinate_new[batch_indices, rc], mic_coordinate_new[batch_indices, nearest_channel] = \
            mic_coordinate[batch_indices, nearest_channel].clone(), mic_coordinate[batch_indices, rc].clone()  

        return x_new, mic_coordinate_new

    def forward(self, x, mic_coordinate, vad=None, target_spherical_position=None, return_target=False):
        """ Input
            x: Multi-channel Waveform (B, C, N)
                B: batch, C: 마이크 채널 수, N: 입력 오디오 샘플 수
            mic_coordinate: 각 마이크의 (x, y, z) 좌표 (B, C, 3)
            vad: 선택 입력; 표본 단위 VAD (B, Q, N)
                Q: 최대 화자 수
            target_spherical_position: 선택 입력; [azimuth, elevation, distance] (B, Q, 3, N)
        """
        x=x.contiguous().to(self.MPE.v.device) # 이후 view()가 안전하게 동작하도록 연속 메모리 배치로 만듦

        # 각 배치별 마이크 좌표 중심: center = mean(mic_coordinate, dim=1) → (B, 1, 3)
        # 중심에 가장 가까운 마이크 index를 찾고, 그 채널을 0번 reference channel과 교환
        x, mic_coordinate=self.ref_channel_modify_to_center(x, mic_coordinate)

        # GCC-PHAT feature
        # GCC_real: (B, P, K, T) = (B, C-1, 257, T)
        # GCC_imag: (B, P, K, T) = (B, C-1, 257, T)
        # 내부 흐름:
        # 1. LNuDFT: x_LNuDFT_real, x_LNuDFT_imag → (B, C, K, T)
        # 2. 기준 채널(0번)과 각 비기준 채널의 cross spectrum 계산
        #    기준 채널은 제거되므로 channel 축이 C → P (= C-1).
        # 3. PHAT normalization 적용
        x_feature = self.input_feature_process(x)
        x_feature=torch.cat(x_feature, dim=2)  # (B, P, 2K, T) = (B, C-1, 514, T)

        # Microphone coordinates processing
        # 기준 채널 좌표를 원점으로 빼고, 기준 채널 자신은 제거
        # MPE: (B, P, F) = (B, C-1, 128)
        # mic_coord_cart: (B, P, 3)
        # mic_coord_dist_sin_cos: (B, P, 5)
        # 마지막 5차원: [sin(azimuth), cos(azimuth), sin(elevation), cos(elevation), distance]
        MPE, mic_coord_cart, mic_coord_dist_sin_cos = self.MPE(mic_coordinate)

        # Channel invariant feature extraction (CIFE)
        # 입력:
        # x_feature:              (B, P, 2K, T) = (B, P, 514, T)
        # MPE:                    (B, P, F)     = (B, P, 128)
        # mic_coord_cart:         (B, P, 3)
        # mic_coord_dist_sin_cos: (B, P, 5)
        #
        # 내부적으로 P를 batch 축에 합침:
        # x_feature    → (B*P, 514, T)
        # 좌표 3+5차원 → (B*P, 8, T)
        #
        # concatenate:
        # (B*P, 514, T) + (B*P, 8, T)
        #     → (B*P, 522, T)
        #
        # ConvBlock 522 → 128 및 4개의 residual block:
        #     → (B*P, 128, T)
        #
        # 마지막에 원래 구조 복원:
        # x_feature: (B, P, F, T) = (B, C-1, 128, T)
        x_feature=self.CIFE(x_feature, MPE, mic_coord_cart, mic_coord_dist_sin_cos)  # (B, P, F, T) = (B, C-1, 128, T)

        # Spatio-temporal dual-path block (STDPB)
        # 공간 경로: 마이크-pair 축 P에 attention
        # x_feature: (B, P, F, T)
        # MPE:       (B, P, F) → (B, P, F, T)
        # x + MPE                                 (B, P, F, T)
        # permute/view for attention               (P, B*T, F)
        # MultiHeadAttention                       (P, B*T, F)
        # 복원                                     (B, P, F, T)

        # 시간 경로: channelwise aggregation 후 GRU
        # channelwise_softmax_aggregation(x, std=True)
        # input: (B, P, F, T), softmax(dim=1): P 축에서 중요도를 계산
        # softmax-weighted sum (B, F, T) + std (B, F, T)를 concatenate → (B, 2F, T)

        # GRU 입력                               (T, B, 2F)
        # GRU 출력                               (T, B, F)
        # 복원                                   (B, 1, F, T)
        # 마지막 1은 residual을 통해 P 축으로 broadcast되어 각 pair feature에 더해짐
        x_spatio_temporal = self.STDPBs(x_feature, MPE)  # (B, P, F, T) = (B, C-1, 128, T)

        # Audio Representation mapping
        # input: (B, P, F, T) = (B, P, 128, T)
        #
        # 먼저 P 축을 softmax-weighted sum + std로 집계:
        # (B, P, F, T) → (B, 2F, T) = (B, 256, T)
        #
        # 3개 residual/mapping stage 각각:
        # residual output: (B, 2F, T) = (B, 256, T)
        # Linear(256 → 256): (B, R, T)
        #
        # 세 stage 출력을 stack:
        # audio_representaion: (B, DS, R, T) = (B, 3, 256, T)
        audio_representaion = self.RMBs(x_spatio_temporal)

        # Probability spatial spectrum estimation
        # 오디오 representation과 고정 Fibonacci 구면 격자를 내적해 공간 스펙트럼 생성
        x_out, DOA_cart_candidates, DOA_spherical_candidates=self.gridnet(audio_representaion)
        # x_out: (B, DS, D, T) = (B, 3, 2048, T), D: fibonacci 격자 점 수
        # DOA_cart_candidates: (D, 3) — fibonacci_sphere로 생성된 고정 구면 격자점의 직교좌표 (x, y, z)
        # DOA_spherical_candidates: (D, 3) — 같은 격자점을 구면좌표로 변환, 마지막 축은 [azimuth, elevation, distance]

        if return_target:
            vad_framed = self.input_feature_process.LNuDFT.get_vad_framed(vad)
            # 입력 vad: (B, Q, N) — 표본(sample) 단위 VAD, Q=최대 화자 수, N=오디오 샘플 수
            # 출력 vad_framed: (B, Q, T) — STFT/NuDFT 프레임 단위 이진 VAD (T=시간 프레임 수, x_out의 T와 동일한 프레임 격자)

            target = target_spatial_spectrum(target_spherical_position, 
                                             DOA_cart_candidates, 
                                             vad_framed, 
                                             self.gammas, 
                                             self.input_feature_process.LNuDFT.get_trajectory_framed)  
            # (B, DS, D, T)

            return x_out, target, DOA_cart_candidates, DOA_spherical_candidates

        else:
            return x_out

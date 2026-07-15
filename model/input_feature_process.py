import torch
from .LNuDFT import LearnableNuDFT

"""
파형을 기준 마이크 대비 GCC-PHAT 복소 특징으로 바꾸는 모듈
"""
class input_feature_process(torch.nn.Module):
    def __init__(self, win_len, ref_channel=0):
        super(input_feature_process, self).__init__()
        self.LNuDFT=LearnableNuDFT(
                            win_len = win_len,
                            hop_len = 128,
                            K = 257, 
                            vad_threshold = 2/3,
                            win_type = 'hann',
                            fs = 16000.0,
                            a_k_trainable = True,
                            eps_0 = 0.01,
                            eps_1 = 100,
                            nu_0 = 0.0,
                            eps_start = 0.15,
                            eps_end = 0.95,
                        )

        self.ref_channel=ref_channel
        self.eps=1e-8    

    def cross_correlation(self, x_real, x_imag, PHAT=True):
        """
        x_real, x_imag: x에 LNuDFT를 적용한 결과 - (B, C, K=257, T)
        """
        # 하나의 복소 텐서로 합침
        comp=torch.complex(x_real, x_imag)

        if PHAT:
            # 복소 스펙트럼의 크기를 1로 정규화하고 위상만 남김
            comp = torch.div(comp, torch.clamp(comp.abs(), min=self.eps))

        comp_ref=comp[:, [self.ref_channel], :, :]  # 기준 마이크만 (B, 1, K, T)
        comp = torch.cat([comp[:, :self.ref_channel, :],       # 기준 채널 앞쪽
                          comp[:, self.ref_channel + 1:, :]],  # 기준 채널 뒤쪽
                          dim=1)                               # (B, C-1, K, T)

        # 각 마이크와 기준 마이크의 cross spectrum
        comp = comp * comp_ref.conj()  # (B, C-1, K, T)
        return comp.real, comp.imag

    def forward(self, x):

        x_LNuDFT_real, x_LNuDFT_imag=self.LNuDFT(x)  # 각각 (B, C, K, T)
        GCC_real, GCC_imag=self.cross_correlation(x_LNuDFT_real, x_LNuDFT_imag, PHAT=True)       

        return GCC_real, GCC_imag  # (B, C-1, K, T)

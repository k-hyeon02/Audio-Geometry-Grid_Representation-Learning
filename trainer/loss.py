from __future__ import annotations

import torch
import torch.nn.functional as F


def spatial_spectrum_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    positive_weight: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AGG-RL A.7의 weighted BCE 손실 (Eq. 22).

    positive(ground-truth DOA) 항에 ``rho`` 가중을 둬서 양성 샘플을 강조한다.
    논문 기본값은 rho = 2.

        L = -(1/(T*D)) * sum_{s,t,d} [ rho * S * log(S_hat)
                                       + (1 - S) * log(1 - S_hat) ]

    output, target: (B, DS, D, T)  (DS = DSCL 출력 수, D = DOA 후보 수, T = 시간 프레임 수)
    각 출력(layer)별 BCE를 따로 계산해 합을 반환하고, layer별 손실도 함께 준다.
    """
    if output.shape != target.shape:
        raise ValueError(
            f"Output/target shape mismatch: {tuple(output.shape)} vs {tuple(target.shape)}"
        )

    eps = 1e-7
    pred = output.clamp(eps, 1.0 - eps)  # (B, DS, D, T)
    tgt = target.float()  # (B, DS, D, T)
    layer_losses = []
    # 논문 Eq.22: positive 항(S·logŜ)에만 rho 가중, negative 항((1-S)·log(1-Ŝ))은 가중 1
    # L = -mean[ rho * S * log(Ŝ) + (1 - S) * log(1 - Ŝ) ]
    # (기존 F.binary_cross_entropy(weight=1+(rho-1)S)는 negative 항까지 곱해져
    #  soft-label peak 주변을 과하게 억눌러 peak가 약해지는 버그가 있었다.)
    elemwise = -(positive_weight * tgt * torch.log(pred) + (1.0 - tgt) * torch.log(1.0 - pred))
    losses = elemwise.mean(dim=(0, 2, 3))  # (DS, )
    return losses.sum(), losses

    # for layer_idx in range(output.shape[1]):
    #     pred = output[:, layer_idx].clamp(eps, 1.0 - eps)
    #     tgt = target[:, layer_idx].float()
    #     # 논문 Eq.22: positive 항(S·logŜ)에만 rho 가중, negative 항((1-S)·log(1-Ŝ))은 가중 1
    #     #   L = -mean[ rho * S * log(Ŝ) + (1 - S) * log(1 - Ŝ) ]
    #     # (기존 F.binary_cross_entropy(weight=1+(rho-1)S)는 negative 항까지 곱해져
    #     #  soft-label peak 주변을 과하게 억눌러 peak가 약해지는 버그가 있었다.)
    #     layer_loss = -(
    #         positive_weight * tgt * torch.log(pred)
    #         + (1.0 - tgt) * torch.log(1.0 - pred)
    #     ).mean()
    #     layer_losses.append(layer_loss)

    # losses = torch.stack(layer_losses)
    # return losses.sum(), losses

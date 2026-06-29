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

        L = -(1/(L*D)) * sum_{o,l,d} [ rho * S * log(S_hat)
                                       + (1 - S) * log(1 - S_hat) ]

    output, target: (B, O, D, L)  (O = DSCL 출력 수)
    각 출력(layer)별 BCE를 따로 계산해 합을 반환하고, layer별 손실도 함께 준다.
    """
    if output.shape != target.shape:
        raise ValueError(
            f"Output/target shape mismatch: {tuple(output.shape)} vs {tuple(target.shape)}"
        )

    layer_losses = []
    for layer_idx in range(output.shape[1]):
        pred = output[:, layer_idx]
        tgt = target[:, layer_idx].float()
        # weight = 1 + (rho - 1) * S  ->  positive(S=1)에서 rho, negative(S=0)에서 1
        # (soft label의 경우 S에 비례해 선형 보간)
        weight = 1.0 + (positive_weight - 1.0) * tgt
        layer_loss = F.binary_cross_entropy(
            pred,
            tgt,
            weight=weight,
            reduction="mean",
        )
        layer_losses.append(layer_loss)

    losses = torch.stack(layer_losses)
    return losses.sum(), losses

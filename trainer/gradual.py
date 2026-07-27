from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpec:
    stage: int
    start_epoch: int
    end_epoch: int
    profile: str
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class ValidationSuiteSpec:
    name: str
    profile: str
    num_samples: int
    channel_schedule: tuple[int, ...] | None = None
    # dynamic 검증 배열은 학습과 동일하게 무작위 3D 회전으로 다양화한다.
    rotate_arrays: bool = True


# AGG-RL Table 6 (MSGL): stage1 = Tetrahedron(4cm, 4ch) epoch 1-10,
# stage2 = dynamic(4ch) epoch 11-20, stage3 = dynamic(4-12ch) epoch 21-300.
# LR/weight_decay도 Table 6을 따른다.
DEFAULT_STAGE_SPECS = (
    StageSpec(1, 1, 10, "stage1", 2.5e-4, 1.0e-4),
    StageSpec(2, 11, 20, "stage2", 5.0e-4, 1.0e-6),
    StageSpec(3, 21, 300, "stage3", 1.0e-3, 1.0e-6),
)

# AGG-RL A.9 (DSCL): beamwidth gamma는 [32, 12, 5]도로 시작해 epoch 35-60에
# 걸쳐 [5, 5, 5]도로 선형 감소한다 (35까지 고정, 60 이후 종료값 유지).
DEFAULT_GAMMA_INIT = (32.0, 12.0, 5.0)
DEFAULT_GAMMA_END = (5.0, 5.0, 5.0)
DEFAULT_CL_FREEZE_UNTIL = 35
DEFAULT_CL_ANNEAL_UNTIL = 60


def stage_for_epoch(epoch: int, specs=DEFAULT_STAGE_SPECS) -> StageSpec:
    for spec in specs:
        if spec.start_epoch <= epoch <= spec.end_epoch:
            return spec
    raise ValueError(f"No stage specification covers epoch {epoch}.")


def total_epochs(specs=DEFAULT_STAGE_SPECS) -> int:
    return max(spec.end_epoch for spec in specs)


def layer_gammas_for_epoch(
    epoch: int,
    num_layers: int,
    gamma_init=DEFAULT_GAMMA_INIT,
    gamma_end=DEFAULT_GAMMA_END,
    freeze_until: int = DEFAULT_CL_FREEZE_UNTIL,
    anneal_until: int = DEFAULT_CL_ANNEAL_UNTIL,
) -> list[float]:
    if num_layers != len(gamma_init) or num_layers != len(gamma_end):
        raise ValueError("num_layers must match the number of curriculum gamma values.")

    if epoch <= freeze_until:
        return list(gamma_init)

    if epoch >= anneal_until:
        return list(gamma_end)

    progress = (epoch - freeze_until) / (anneal_until - freeze_until)
    return [
        float(start + progress * (end - start))
        for start, end in zip(gamma_init, gamma_end)
    ]


def validation_suite_for_stage(
    stage: int,
    fixed_suite_samples: int = 2_000,
    dynamic_samples_per_channel: int = 300,
) -> ValidationSuiteSpec:
    """AGG-RL A.9의 stage별 검증 스위트를 하나 반환한다.

    논문 A.9: "validation was conducted after every epoch
    (2,000 samples in stages 1–2, and 300 samples per channel in stage 3)".
    검증셋은 학습과 동일하게 test-clean + MS-SNSD test로 합성 생성하며,
    배열 프로필은 해당 stage의 학습 프로필(Table 6)과 일치시킨다.
    stage 1 = Tetrahedron(4ch), stage 2 = dynamic(4ch),
    stage 3 = dynamic(4-12ch, 채널당 300 샘플).
    """
    if stage == 1:
        # stage 1 학습 배열(Tetrahedron 4ch)과 동일한 프로필로 2,000 샘플 검증.
        return ValidationSuiteSpec("stage1", "stage1", fixed_suite_samples)
    if stage == 2:
        # stage 2 학습 배열(dynamic 4ch)과 동일한 프로필로 2,000 샘플 검증.
        return ValidationSuiteSpec("stage2", "stage2", fixed_suite_samples)
    if stage == 3:
        # stage 3: dynamic 4-12ch, 채널당 dynamic_samples_per_channel 샘플.
        channel_schedule = tuple(
            channel
            for channel in range(4, 13)
            for _ in range(dynamic_samples_per_channel)
        )
        return ValidationSuiteSpec(
            "stage3",
            "stage3",
            len(channel_schedule),
            channel_schedule=channel_schedule,
        )
    raise ValueError(f"No validation suite defined for stage {stage}.")

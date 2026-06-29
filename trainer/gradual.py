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
    # 고정 상용 배열(nao12 등)은 native orientation으로 평가한다.
    # 평면형 배열을 무작위 회전시키면 azimuth 기준계가 틀어지므로,
    # 무작위 3D 회전은 dynamic 배열 다양화에만 쓴다.
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


def default_validation_suites(
    fixed_suite_samples: int = 2_000,
    dynamic_samples_per_channel: int = 300,
) -> tuple[ValidationSuiteSpec, ...]:
    # AGG-RL A.9: stage 1-2 검증은 2,000 샘플, stage 3는 채널당 300 샘플.
    # 검증 배열은 nao12(seen 실데이터 NAO robot 대응) + dynamic(4ch / 4-12ch).
    # GI-DOAEnet이 쓰던 respeaker 스위트는 AGG-RL에서 사용하지 않으므로 제외한다.
    channel_schedule = tuple(
        channel
        for channel in range(4, 13)
        for _ in range(dynamic_samples_per_channel)
    )
    return (
        ValidationSuiteSpec("nao12", "nao12", fixed_suite_samples, rotate_arrays=False),
        ValidationSuiteSpec("dynamic4", "dynamic4", fixed_suite_samples),
        ValidationSuiteSpec(
            "dynamic4to12",
            "dynamic4to12",
            len(channel_schedule),
            channel_schedule=channel_schedule,
        ),
    )

from __future__ import annotations

import argparse

import torch

from data.dataset import SyntheticDOADataset
from data.simulate import SimulationConfig
from model.main import AGG_RL
from trainer.gradual import DEFAULT_STAGE_SPECS, default_validation_suites
from trainer.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AGG-RL with CGT (MSGL + DSCL).")
    parser.add_argument("--librispeech_train", required=True)  # train-clean-100
    parser.add_argument("--librispeech_val", required=True)    # test-clean
    parser.add_argument("--ms_snsd_train", required=True)
    parser.add_argument("--ms_snsd_val", required=True)
    # AGG-RL 기본은 FM-based rMPE (4.1: "FM-based rMPE was adopted as default").
    parser.add_argument("--mpe_type", choices=["FM", "PM"], default="FM")
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--batch_size", type=int, default=16)  # A.9: batch size 16
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=0)
    # A.9: "Each epoch contained 28,800 utterances for training"
    parser.add_argument("--train_samples_per_epoch", type=int, default=28_800)
    # A.9: 검증은 stage 1-2에서 2,000 샘플, stage 3에서 채널당 300 샘플
    parser.add_argument("--fixed_suite_samples", type=int, default=2_000)
    parser.add_argument("--dynamic_samples_per_channel", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    simulation_config = SimulationConfig()

    # AGG-RL 모델 생성 (LNuDFT + rMPE + AuGeonet + Gridnet)
    model = AGG_RL(MPE_type=args.mpe_type).to(device)

    # 학습 데이터셋: stage 1(Tetrahedron 4ch)로 시작, trainer가 stage별로 set_profile 전환.
    # on-the-fly 생성이라 매 에폭 set_epoch로 새 샘플을 만든다.
    train_dataset = SyntheticDOADataset(
        librispeech_root=args.librispeech_train,
        ms_snsd_root=args.ms_snsd_train,
        num_samples=args.train_samples_per_epoch,
        profile="stage1",
        batch_size=args.batch_size,
        seed=args.seed,
        simulation_config=simulation_config,
    )

    # 검증 suite들(nao12 / dynamic4 / dynamic4to12)을 각각 데이터셋으로 구성
    validation_datasets = {}
    for suite_index, suite in enumerate(
        default_validation_suites(
            fixed_suite_samples=args.fixed_suite_samples,
            dynamic_samples_per_channel=args.dynamic_samples_per_channel,
        ),
        start=1,
    ):
        validation_datasets[suite.name] = SyntheticDOADataset(
            librispeech_root=args.librispeech_val,
            ms_snsd_root=args.ms_snsd_val,
            num_samples=suite.num_samples,
            profile=suite.profile,
            batch_size=args.batch_size,
            seed=args.seed + 1_000 * suite_index,
            simulation_config=simulation_config,
            channel_schedule=suite.channel_schedule,
            rotate_arrays=suite.rotate_arrays,
        )

    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device}")
    print(f"model=AGG_RL({args.mpe_type}) params={num_params:,}")
    print(
        f"train_speech={len(train_dataset.speech_files)} "
        f"train_noise={len(train_dataset.noise_files)}"
    )
    for name, dataset in validation_datasets.items():
        print(
            f"val_suite={name} speech={len(dataset.speech_files)} "
            f"noise={len(dataset.noise_files)} samples={len(dataset)}"
        )

    train(
        model=model,
        train_dataset=train_dataset,
        validation_datasets=validation_datasets,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        resume_path=args.resume,
        max_epochs=args.max_epochs,
        stage_specs=DEFAULT_STAGE_SPECS,
    )


if __name__ == "__main__":
    main()

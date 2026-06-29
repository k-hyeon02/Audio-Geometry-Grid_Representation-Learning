from __future__ import annotations

import os
import time
from typing import Mapping

import torch
from torch.optim import Adam

from data.dataset import SyntheticDOADataset, build_dataloader
from .gradual import (
    DEFAULT_STAGE_SPECS,
    layer_gammas_for_epoch,
    stage_for_epoch,
    total_epochs,
)
from .loss import spatial_spectrum_loss


def _set_optimizer_stage(optimizer: Adam, learning_rate: float, weight_decay: float) -> None:
    # MSGL stage 전환 시 optimizer의 LR/weight_decay를 stage 값으로 교체 (Table 6).
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
        group["weight_decay"] = weight_decay


class ValidationPlateauScheduler:
    # AGG-RL A.9: 검증 손실이 patience 에폭 연속 개선되지 않으면 LR을 factor배 감쇠.
    # 논문 설정은 factor=0.9, patience=2.
    def __init__(
        self,
        optimizer: Adam,
        factor: float = 0.9,
        patience: int = 2,
    ) -> None:
        self.optimizer = optimizer
        self.factor = float(factor)
        self.patience = int(patience)
        self.best = float("inf")   # 지금까지 가장 낮은 검증 손실
        self.bad_epochs = 0        # 연속으로 개선 못한 에폭 수

    def step(self, metric: float) -> bool:
        metric = float(metric)
        # 개선되면 best 갱신하고 카운터 리셋
        if metric < self.best:
            self.best = metric
            self.bad_epochs = 0
            return False

        # 개선 못했으면 카운터 증가, patience 미만이면 아직 감쇠 안 함
        self.bad_epochs += 1
        if self.bad_epochs < self.patience:
            return False

        # patience만큼 정체되면 모든 param group의 LR을 factor배로 줄임
        for group in self.optimizer.param_groups:
            group["lr"] *= self.factor
        self.bad_epochs = 0
        return True

    def state_dict(self) -> dict[str, float | int]:
        return {
            "factor": self.factor,
            "patience": self.patience,
            "best": self.best,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state_dict: Mapping[str, float | int]) -> None:
        self.factor = float(state_dict.get("factor", self.factor))
        self.patience = int(state_dict.get("patience", self.patience))
        self.best = float(state_dict.get("best", self.best))
        self.bad_epochs = int(state_dict.get("bad_epochs", self.bad_epochs))


def _run_loader(
    model: torch.nn.Module,
    loader,
    device: str,
    gammas: list[float],
    optimizer: Adam | None = None,
    grad_clip: float | None = None,
    log_label: str | None = None,
    log_interval: int = 0,
) -> tuple[float, torch.Tensor, int]:
    # optimizer가 주어지면 학습, 아니면 검증(no backward)으로 한 에폭(또는 한 suite)을 돈다.
    total_loss = 0.0
    total_layers = torch.zeros(len(gammas), dtype=torch.float64)  # DSCL 출력별 손실 누적
    num_batches = 0
    start_time = time.monotonic()
    total_batches = len(loader) if hasattr(loader, "__len__") else None

    for batch in loader:
        # 배치 텐서를 모두 학습 디바이스로 이동
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        input_audio = batch["input_audio"]
        vad = batch["vad"]
        mic_coordinate = batch["mic_coordinate"]
        # AGG-RL은 시간축 trajectory(spherical_position)로 oracle target을 만든다.
        # (GI-DOAEnet은 정적 polar_position을 썼지만, AGG-RL은 이동 음원까지 다루므로
        #  시간축으로 framing되는 spherical_position을 입력으로 받는다.)
        spherical_position = batch["spherical_position"]

        # 현재 에폭의 DSCL 커리큘럼 gamma를 모델에 주입.
        # AGG_RL.forward(return_target=True)가 내부에서 LNuDFT 기반 VAD framing,
        # trajectory framing, Fibonacci grid oracle spectrum까지 만들어
        # (pred, target)을 함께 돌려준다 (model/main.py:106-111 참고).
        model.gammas = gammas
        output, target, _, _ = model(
            input_audio,
            mic_coordinate,
            vad=vad,
            target_spherical_position=spherical_position,
            return_target=True,
        )
        # 논문 A.7의 weighted BCE(rho=2)로 출력별 손실 계산
        loss, layer_losses = spatial_spectrum_loss(output, target)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # A.9: gradient clipping within 1
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.item())
        total_layers += layer_losses.detach().cpu().to(torch.float64)
        num_batches += 1
        # 일정 간격마다 진행 상황 로깅
        if log_label and log_interval > 0 and num_batches % log_interval == 0:
            elapsed = time.monotonic() - start_time
            total_text = f"/{total_batches}" if total_batches is not None else ""
            print(
                f"[{time.strftime('%H:%M:%S')}] {log_label} "
                f"batch={num_batches}{total_text} "
                f"avg_loss={total_loss / num_batches:.4f} "
                f"elapsed_min={elapsed / 60.0:.1f}",
                flush=True,
            )

    # 배치가 하나도 없으면(빈 loader) 0으로 반환
    if num_batches == 0:
        return 0.0, total_layers, 0
    return total_loss / num_batches, total_layers / num_batches, num_batches


def train(
    model: torch.nn.Module,
    train_dataset: SyntheticDOADataset,
    validation_datasets: Mapping[str, SyntheticDOADataset],
    device: str,
    batch_size: int = 16,
    num_workers: int = 4,
    prefetch_factor: int = 1,
    log_interval: int = 0,
    save_dir: str = "./checkpoints",
    resume_path: str | None = None,
    stage_specs=DEFAULT_STAGE_SPECS,
    max_epochs: int | None = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # 초기 stage(stage 1)의 LR/weight_decay로 Adam 옵티마이저 생성 (A.9: Adam).
    initial_stage = stage_specs[0]
    optimizer = Adam(
        model.parameters(),
        lr=initial_stage.learning_rate,
        weight_decay=initial_stage.weight_decay,
    )
    scheduler = ValidationPlateauScheduler(
        optimizer,
        factor=0.9,
        patience=2,
    )

    start_epoch = 1
    best_val_loss = float("inf")

    # resume 체크포인트가 있으면 모델/옵티마이저/스케줄러 상태를 복원하고 이어서 학습
    if resume_path is not None and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))

    # 검증 데이터셋은 에폭에 무관하게 고정된 분포를 쓰도록 epoch 0으로 고정
    for dataset in validation_datasets.values():
        dataset.set_epoch(0)

    # 전체 학습 종료 에폭(기본 300). max_epochs가 주어지면 그만큼만 추가로 돈다.
    end_epoch = total_epochs(stage_specs)
    if max_epochs is not None:
        end_epoch = min(end_epoch, start_epoch + max_epochs - 1)

    current_stage = None
    for epoch in range(start_epoch, end_epoch + 1):
        # 현재 에폭이 속한 MSGL stage를 찾고, stage가 바뀌면 LR/weight_decay/프로필 전환
        stage = stage_for_epoch(epoch, stage_specs)
        if current_stage != stage.stage:
            _set_optimizer_stage(
                optimizer,
                learning_rate=stage.learning_rate,
                weight_decay=stage.weight_decay,
            )
            train_dataset.set_profile(stage.profile)
            current_stage = stage.stage

        # 에폭마다 데이터 시드를 갱신해 on-the-fly로 새 샘플을 생성
        train_dataset.set_epoch(epoch)
        train_loader = build_dataloader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            shuffle=True,
        )

        # 현재 에폭의 DSCL beamwidth gamma (coarse->fine)
        gammas = layer_gammas_for_epoch(
            epoch=epoch,
            num_layers=len(model.gammas),
        )

        # ---- 학습 ----
        model.train()
        train_loss, train_layer_loss, _ = _run_loader(
            model=model,
            loader=train_loader,
            device=device,
            gammas=gammas,
            optimizer=optimizer,
            grad_clip=1.0,
            log_label=f"epoch={epoch:03d} train",
            log_interval=log_interval,
        )

        # ---- 검증 (suite별) ----
        model.eval()
        validation_results: dict[str, float] = {}
        weighted_val_loss = 0.0
        weighted_batches = 0
        with torch.no_grad():
            for name, dataset in validation_datasets.items():
                val_loader = build_dataloader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    shuffle=False,
                )
                val_loss, _, num_batches = _run_loader(
                    model=model,
                    loader=val_loader,
                    device=device,
                    gammas=gammas,
                    optimizer=None,
                    log_label=f"epoch={epoch:03d} val:{name}",
                    log_interval=log_interval,
                )
                validation_results[name] = val_loss
                # suite별 배치 수로 가중 평균 (큰 suite가 더 큰 영향)
                weighted_val_loss += val_loss * num_batches
                weighted_batches += num_batches

        # 전체 검증 손실(가중 평균)로 LR 스케줄러 갱신
        mean_val_loss = weighted_val_loss / max(weighted_batches, 1)
        scheduler.step(mean_val_loss)

        # 에폭 요약 로그
        timestamp = time.strftime("%H:%M:%S")
        suite_summary = ", ".join(
            f"{name}={loss:.4f}" for name, loss in validation_results.items()
        )
        print(
            f"[{timestamp}] epoch={epoch:03d} stage={stage.stage} "
            f"gamma={[round(g, 2) for g in gammas]} "
            f"train={train_loss:.4f} val={mean_val_loss:.4f} "
            f"layers={[round(float(x), 4) for x in train_layer_loss.tolist()]} "
            f"val_suites({suite_summary})"
        )

        # 체크포인트 저장: 매 에폭 latest, 최저 검증손실 best, 10에폭마다 스냅샷
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "stage": stage.stage,
            "gammas": gammas,
        }
        torch.save(checkpoint, os.path.join(save_dir, "latest.pt"))

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, os.path.join(save_dir, "best.pt"))

        if epoch % 10 == 0:
            torch.save(checkpoint, os.path.join(save_dir, f"epoch_{epoch:04d}.pt"))

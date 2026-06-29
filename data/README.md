# Dataset pipeline (ported from GI-DOAEnet)

This `data/` package is **GI-DOAEnet's data code** (`/home/s2021102349/GI-DOAEnet-main/data/`)
ported here and adapted to the AGG-RL (ICLR 2026) input format. GI-DOAEnet
already had a complete, training-validated synthetic pipeline, so it is reused
rather than rewritten.

## Files

| File | Origin | Change |
|---|---|---|
| `simulate.py` | GI-DOAEnet | + `spherical_position (max_spk,3,T)` output; vMF elevation (A.10) |
| `mic_arrays.py` | GI-DOAEnet | + `TETRAHEDRON_4CM` array (Table 6 stage 1) |
| `dataset.py` | GI-DOAEnet | emits `spherical_position`; stage1 profile → tetrahedron |
| `__init__.py` | GI-DOAEnet | export `TETRAHEDRON_4CM` |
| `../generate_dataset.py` | new | dump `sample_input`-format `.pkl` for `inference.py` |

## What was adapted for AGG-RL

GI-DOAEnet and AGG-RL share the corpora (LibriSpeech / MS-SNSD / TIMIT / ESC-50),
gpuRIR simulation, MSGL channel curriculum, and Eq. (23) distance bounds. Three
differences were bridged:

1. **Position label key & shape.** GI-DOAEnet outputs `polar_position (max_spk, 3)`;
   AGG-RL's `inference.py` consumes `spherical_position (max_spk, 3, T)` (the model
   frames it over time via `LearnableNuDFT.get_trajectory_framed` for moving
   sources). Sources are static here, so the per-speaker `[az, el, dist]` is
   broadcast across the time axis. **Both keys are emitted** for compatibility.

2. **Elevation distribution.** AGG-RL A.10 draws elevation from
   `vMF(mu=pi, kappa=2)` then halves it (biasing toward the horizontal plane);
   GI-DOAEnet used `uniform(30, 150)`. Controlled by
   `SimulationConfig.elevation_use_vmf` (default `True`); set `False` to restore
   the GI-DOAEnet behaviour.

3. **Stage-1 array.** AGG-RL Table 6 uses a 4 cm tetrahedron; GI-DOAEnet used the
   ReSpeaker array. Profile `stage1`/`tetrahedron` now selects `TETRAHEDRON_4CM`
   (verified at exactly 4.00 cm edge). `respeaker` profile is still available.

VAD `(n_spk, T)` and `mic_coordinate (C, 3)` already matched.

## Output schema (matches `sample_input/<C>/<idx>.pkl`)

```
vad                (n_spk, T)        bool
n_spk              int
n_channel          int
mic_dim            str   ('D3')
input_audio        (C, T)            float32
mic_coordinate     (C, 3)            float64
spherical_position (n_spk, 3, T)     float64   # [azimuth°, elevation°, distance_m]
```

## Install

Same deps as GI-DOAEnet's data pipeline: `gpuRIR`, `soundfile`, `webrtcvad`,
`scipy`, `pandas`, `torch` (gpuRIR needs a CUDA toolkit to build).

## Usage

### Dump eval `.pkl` files (runs through `inference.py`)

```sh
python generate_dataset.py \
    --librispeech_root /path/LibriSpeech/test-clean \
    --ms_snsd_root     /path/MS-SNSD/noise_test \
    --out_dir ./generated --channels 4 8 12 --per_channel 10 --array dynamic
```

### On-the-fly training dataset

```python
from data.dataset import SyntheticDOADataset, build_dataloader
from data.simulate import SimulationConfig

ds = SyntheticDOADataset(
    librispeech_root="/path/LibriSpeech/train-clean-100",
    ms_snsd_root="/path/MS-SNSD/noise_train",
    num_samples=28800, profile="stage1", batch_size=16,
    simulation_config=SimulationConfig(),
)
loader = build_dataloader(ds, batch_size=16, num_workers=4, shuffle=True)

for epoch in range(1, 301):
    ds.set_epoch(epoch)
    ds.set_profile("stage1" if epoch <= 10 else "stage2" if epoch <= 20 else "stage3")
    for batch in loader:
        pred, target, *_ = model(
            batch["input_audio"], batch["mic_coordinate"],
            vad=batch["vad"], target_spherical_position=batch["spherical_position"],
            return_target=True)
        # loss = weighted_bce(pred, target)   # Eq. 22, rho=2
```

> `ChannelGroupBatchSampler` (used by `build_dataloader`) keeps each batch at a
> single channel count, as the paper requires.

## Not ported

GI-DOAEnet's `trainer/` (loss, gradual training, train loop) is **not** copied
here — its loss/output head differs from AGG-RL's AGG-RL similarity + DSCL. Ask
to port/adapt the trainer next.

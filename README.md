# SDPO vs GRPO — Qwen3.5-9B on tooluse

Comparison of **SDPO** ([Hübotter et al., 2026](https://arxiv.org/abs/2601.20802))
against **GRPO** for full-fine-tuning **Qwen/Qwen3.5-9B** on the `tooluse` dataset
(multi-turn ReAct-style tool calling, 4046 train / 68 test).

Runs both methods through the official reference code at
[`lasgroup/SDPO`](https://github.com/lasgroup/SDPO) (a verl fork) — same
framework, same hyperparameters, only the loss differs. Launched on a single
Prime Intellect B200 180GB pod.

## What's here

```
.
├── upstream/                      # git submodule -> lasgroup/SDPO
├── configs/
│   ├── grpo.sh                    # baseline_grpo + Qwen3.5-9B + tooluse
│   └── sdpo.sh                    # sdpo + Qwen3.5-9B + tooluse (alpha=0.5, JS divergence)
├── prepare.sh                     # preprocess tooluse to parquet (one-time)
├── prime/
│   ├── launch.sh                  # local-side: rent B200, ssh, run, sync, terminate
│   └── pod_setup.sh               # pod-side: install deps, prepare data, run both methods
└── results/                       # populated after runs (W&B JSON dumps + plots)
```

## Hardware

| Stage | GPU | Hourly | Est. wall-clock |
|---|---|---|---|
| GRPO run | 1× H200 141GB (datacrunch FI, id `468e80`) | $1.187 | ~3 hr |
| SDPO run | 1× H200 141GB (datacrunch FI, id `468e80`) | $1.187 | ~3 hr |
| **Total** | | | **~$8–10 / 6–8 hr** |

GPU memory at 141GB is tight for 9B full-FT, so configs enable
`fsdp_config.optimizer_offload=True` (AdamW states in CPU RAM, ~72 GB
saved on GPU) and `use_dynamic_bsz=True`. Falls back to 8× A100
($5.52/hr, 640GB total) if it OOMs even with offload.

## Hyperparameters (paper defaults, single-GPU)

Identical between GRPO and SDPO except where noted:

```
train_batch_size       = 32
rollout.n              = 8           # 8 completions per prompt
ppo_mini_batch_size    = 8           # GRPO  /  32 for SDPO (matches paper sweep)
learning_rate          = 1e-5
lr_warmup_steps        = 10
val_kwargs.n           = 16          # 16-sample eval
N_GPUS_PER_NODE        = 1
```

SDPO-only:
```
self_distillation.alpha               = 0.5    # generalized JS divergence
self_distillation.distillation_topk   = 100
self_distillation.is_clip             = 2.0
dont_reprompt_on_self_success         = True
algorithm.rollout_correction.rollout_is = "token"
```

## Running it

```bash
# 1) one-time: clone with submodule
git clone --recurse-submodules https://github.com/yug-space/sdpo-vs-grpo
cd sdpo-vs-grpo

# 2) preflight: log in to Prime, check wallet
prime login
prime --plain wallet

# 3) launch (from local laptop) — provisions pod, runs both methods, syncs results back
bash prime/launch.sh
```

## Why tooluse and not GSM8K / OSWorld

- **GSM8K**: easier to verify but a saturated benchmark; SDPO's effect over GRPO
  on GSM8K is in the noise per the paper.
- **OSWorld / AndroidWorld**: needs multimodal model (Qwen3.5-9B is text-only)
  and a display environment — out of scope for this budget.
- **tooluse**: paper's own default for the no-rich-feedback setting; the SDPO
  paper reports its largest GRPO→SDPO gap on this kind of task (Figure 1 right).

## Results

Populated after the run:
- `results/grpo_metrics.json` — train/eval/val curves
- `results/sdpo_metrics.json` — same
- `results/comparison.png` — overlaid avg@16 vs steps

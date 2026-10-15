"""Experiment settings. Override any field from the command line (see scripts)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # Model. `unsloth/gemma-3-1b-it` is an ungated mirror if you skip the HF licence step.
    model_id: str = "google/gemma-3-1b-it"
    device: str = "auto"          # auto -> mps / cuda / cpu
    dtype: str = "auto"           # auto -> bfloat16 on mps/cuda, float32 on cpu

    # Layers (Gemma 3 1B has 26). Gemma Scope 2 1B-it SAEs exist at 7, 13, 17, 22.
    inject_layer: int = 7         # activation body (condition G) is added here
    probe_layers: tuple = (13, 17)
    loop_layer: int = 13          # probe layer whose threat score drives the closed loop

    # SAE (optional; needs ~150 MB per layer)
    sae_release: str = "gemma-scope-2-1b-it-res"
    sae_id_template: str = "layer_{layer}_width_16k_l0_medium"
    sae_layer: int = 13
    sae_top_k: int = 10

    # Generation
    max_new_tokens: int = 60
    history_turns: int = 6        # keep the system prompt + last N exchanges (speed; 0 = keep all)
    temperature: float = 0.8

    # Closed loop
    loop_gains: tuple = (0.05, 0.1, 0.2)

    # Activation body: steering strength as a multiple of the typical activation norm
    inject_strength: float = 0.6

    # Paths
    probe_dir: str = "artifacts/probes"
    run_dir: str = "runs"

    emotion_words: tuple = ("afraid", "anxious", "excited", "ready", "calm", "tired")
    channels: tuple = ("heart", "breath", "gut", "energy")


def from_args(args) -> Config:
    c = Config()
    for k, v in vars(args).items():
        if v is not None and hasattr(c, k):
            setattr(c, k, v)
    return c
